import numpy as np
import trimesh
import mujoco
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as sRot
import torch

Q_UPRIGHT_XYZW = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
Rupright = sRot.from_quat(Q_UPRIGHT_XYZW)

GRAVITY = torch.tensor([0.0, 0.0, -9.81])  # z 轴向上坐标系
AIRBORNE_H = 0.05        # 空中判定高度阈值（m）
ACC_DEV_THR = 2.0        # “偏离重力”判定阈值（m/s^2）
GROUND_H_TOL = 0.02      # 认为在地面的 z 高度容差（m）
SPEED_THR = 0.05         # 地面时“仍在运动”的速度阈值（m/s）
CLOSE_MIN_THR = 0.05     # 最近距离判定交互阈值（m）
ADAPTIVE_PAD = 0.005     # sigma_t = min_dist + 0.005（m）
FIXED_SIGMA_NO_INTERACT = 0.02  # 若不满足三条交互条件时的保底阈值（可选）


# ========== 1) 从 XML 构建每个 geom 的“局部模板点云” ==========
def build_local_templates_from_xml(xml_path, samples_per_geom=1500):
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # 收集 mesh 几何的顶点/三角面（局部坐标）
    mesh_trimesh = []
    for m in range(mj_model.nmesh):
        adr = mj_model.mesh_vertadr[m]
        n  = mj_model.mesh_vertnum[m]
        verts = mj_model.mesh_vert[adr:adr+n].copy()         # (n,3)
        fadr = mj_model.mesh_faceadr[m]
        fn   = mj_model.mesh_facenum[m]
        faces = mj_model.mesh_face[fadr:fadr+fn].copy()      # (fn,3) int32
        mesh_trimesh.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

    local_clouds = []     # list of (points_local [K,3], body_id, geom_id)

    geom2body = mj_model.geom_bodyid.copy()
    geom_type = mj_model.geom_type.copy()
    geom_size = mj_model.geom_size.copy()

    for g in range(mj_model.ngeom):
        gtype = geom_type[g]
        size  = geom_size[g]
        points_local = None

        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            tm = trimesh.creation.icosphere(subdivisions=3, radius=r)
            points_local, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            # MuJoCo: box size 是半尺寸，局部对齐轴：X/Y/Z
            half = size[:3].astype(float)
            tm = trimesh.creation.box(extents=half*2.0)
            points_local, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            # ✅ MuJoCo 胶囊默认沿 **Z 轴**，height = 2*half
            r = float(size[0]); half = float(size[1])
            tm = trimesh.creation.capsule(radius=r, height=2.0*half, count=[32, 8])
            # ❌ 不要再旋到 X 轴了：
            # tm.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
            points_local, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # ✅ MuJoCo 圆柱也默认沿 **Z 轴**
            r = float(size[0]); half = float(size[1])
            tm = trimesh.creation.cylinder(radius=r, height=2.0*half, sections=48)
            # ❌ 不要旋转到 X 轴：
            # tm.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0]))
            points_local, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = mj_model.geom_dataid[g]  # mesh id
            tm = mesh_trimesh[mid]
            points_local, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        else:
            # 其他类型（plane/heightfield/ellipsoid 等）一般不是身体碰撞体
            continue

        local_clouds.append((points_local.astype(np.float32), int(geom2body[g]), g))

    return mj_model, local_clouds


# ========== 2) 单帧：用 mj_forward 得到世界系点云（按 body 聚合） ==========
def worldize_clouds_per_frame(mj_model, mj_data, local_clouds):
    """
    返回:
      body_clouds: dict {body_id: np.ndarray [M,3]}
    """
    body_clouds = {}
    # MuJoCo 给每个 geom 世界位姿：geom_xpos[g] 和 geom_xmat[g] (3x3)
    for pts_local, body_id, g in local_clouds:
        xpos = mj_data.geom_xpos[g].copy()         # (3,)
        xmat = mj_data.geom_xmat[g].reshape(3,3)   # (9,) -> (3,3)
        # 刚体变换
        pts_world = (pts_local @ xmat.T) + xpos[None, :]
        if body_id not in body_clouds:
            body_clouds[body_id] = pts_world
        else:
            body_clouds[body_id] = np.vstack([body_clouds[body_id], pts_world])
    return body_clouds

from tqdm import tqdm

def contacts_from_xml_pointcloud(
    mj_model, local_clouds,
    qpos_seq,                  # [T, nq]
    obj_pts_world,             # [T, P, 3]  numpy
    obj_pos, vel, acc,         # [T,3]      numpy
    sigma_pad=0.005, sigma_no_interact=0.02,
    ground_height=0.0,
    show_progress=True,
    progress_desc="contacts",
    workers=1                  # cKDTree.query 的并行线程；1 更稳，-1 可能更快
):
    T = qpos_seq.shape[0]
    contact = np.zeros((T, 52), dtype=np.float32)
    mj_data = mujoco.MjData(mj_model)

    # 一次性把重力转成 numpy，避免循环内反复 .cpu()
    try:
        gravity_np = GRAVITY.cpu().numpy()
    except Exception:
        gravity_np = np.array([0.0, 0.0, -9.81], dtype=np.float32)

    frame_iter = range(T)
    if show_progress:
        frame_iter = tqdm(frame_iter, desc=progress_desc, leave=False)

    for t in frame_iter:
        # — 前向到第 t 帧 —
        mj_data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(mj_model, mj_data)

        # — 聚合本帧 body 点云 —
        body_clouds = worldize_clouds_per_frame(mj_model, mj_data, local_clouds)
        if not body_clouds:
            continue

        # —— 单树单次查询：拿 d_all & 各 body 的 min 距离 —— #
        (dmin, nearest_body, d_all), min_row = _min_dist_per_body_single_tree(
            obj_pts_world[t], body_clouds, n_bodies=52, workers=workers
        )

        # — 触发条件 + 自适应阈值 —
        airborne = (obj_pos[t, 2] - ground_height) > AIRBORNE_H
        acc_dev  = np.linalg.norm(acc[t] - gravity_np) > ACC_DEV_THR
        near_g   = abs(obj_pos[t, 2] - ground_height) <= GROUND_H_TOL
        moving   = np.linalg.norm(vel[t]) > SPEED_THR
        interacted = (airborne and acc_dev) or (near_g and moving) or (d_all < CLOSE_MIN_THR)
        sigma_t = float(d_all + sigma_pad) if interacted else float(sigma_no_interact)

        # — 判阈值得到 contact[t] —
        contact[t, :] = (min_row < sigma_t).astype(np.float32)

    return contact


def build_qpos_seq_from_state(mj_model, new_sk_state):
    """
    生成 [T, nq] 的 qpos 序列，匹配你在 vis_mujoco 里的约定：
      qpos[:3]   = root_trans
      qpos[3:7]  = root_quat (xyzw)
      qpos[7:]   = 每个关节以 3 维轴角展开并平铺（与 xml 里每关节 3 个 hinge 对齐）
    注意：这假定 xml 的非 root 关节是 3 个 hinge（x/y/z）次序与 SkeletonTree 一致。
    """
    T = new_sk_state.root_translation.shape[0]
    nq = mj_model.nq
    qpos_seq = np.zeros((T, nq), dtype=np.float32)

    # root
    root_trans = new_sk_state.root_translation.numpy().astype(np.float32)           # [T,3]
    root_quat_wxyz = new_sk_state.global_root_rotation.numpy().astype(np.float32)   # [T,4] wxyz
    root_quat_xyzw = root_quat_wxyz[..., [1,2,3,0]]                                 # -> xyzw

    qpos_seq[:, :3] = root_trans
    qpos_seq[:, 3:7] = root_quat_xyzw

    # 其余关节：把 local_rotation[:,1:] (除根) 的四元数转成轴角，然后平铺
    #   这与你 vis_mujoco 里:
    #   mj_data.qpos[7:] = sRot.from_quat(new_sk_state.local_rotation[:,1:]...).as_rotvec().reshape(N, -1, 3)
    #   一致
    local_quat = new_sk_state.local_rotation[:, 1:].reshape(T, -1, 4).numpy()   # [T, (J-1), 4]
    dof_axis_angle = sRot.from_quat(local_quat.reshape(-1, 4)).as_rotvec().reshape(T, -1)  # [T, 3*(J-1)]
    assert qpos_seq.shape[1] == 7 + dof_axis_angle.shape[1], \
        f"nq mismatch: model.nq={qpos_seq.shape[1]}, got dof length={7 + dof_axis_angle.shape[1]}"
    qpos_seq[:, 7:] = dof_axis_angle.astype(np.float32)
    return qpos_seq

from scipy.spatial import cKDTree

def _min_dist_per_body_single_tree(obj_pts, body_clouds_dict, n_bodies=52, workers=1):
    """
    obj_pts: [P,3]  物体表面点（世界系）
    body_clouds_dict: {body_id -> np.ndarray [Mi,3]}  本帧各 body 的点云
    return:
      contact_inputs:
        dmin: [P]           每个物体点到机器人最近点的距离
        nearest_body: [P]   最近点所属 body
        d_all: float        全局最小距离 min(dmin)
      min_dist_row: [52]    各 body 的最小距离（尚未判阈值）
    """
    chunks, owners = [], []
    for b, pts in body_clouds_dict.items():
        if pts is None or pts.size == 0:
            continue
        chunks.append(pts)
        owners.append(np.full((pts.shape[0],), b, dtype=np.int32))
    if not chunks:
        # 没有可用点
        dmin = np.full((obj_pts.shape[0],), np.inf, dtype=np.float32)
        nearest_body = np.full((obj_pts.shape[0],), -1, dtype=np.int32)
        min_dist = np.full((n_bodies,), np.inf, dtype=np.float32)
        return (dmin, nearest_body, np.inf), min_dist

    ALL = np.vstack(chunks)          # [M,3]
    OWN = np.concatenate(owners)     # [M]
    tree = cKDTree(ALL)
    dmin, idx = tree.query(obj_pts, k=1, workers=workers)   # 建一次树、查一次
    nearest_body = OWN[idx]                                  # [P]
    d_all = float(dmin.min())

    # 按 body 聚合最小距离
    min_dist = np.full((n_bodies,), np.inf, dtype=np.float32)
    for b in range(n_bodies):
        mask = (nearest_body == b)
        if not np.any(mask):
            continue
        md = float(dmin[mask].min())
        min_dist[b] = md

    return (dmin.astype(np.float32), nearest_body.astype(np.int32), d_all), min_dist
def _bounding_sphere(points):
    c = points.mean(axis=0)
    r = np.sqrt(((points - c)**2).sum(axis=1)).max()
    return c, r

def _prune_bodies_by_sphere(obj_pts, body_clouds_dict, sigma_t, margin=0.01):
    keep = {}
    for b, pts in body_clouds_dict.items():
        if pts.size == 0:
            continue
        c, r = _bounding_sphere(pts)
        # 物体点到球心的最小距离
        dmin = np.sqrt(((obj_pts - c[None, :])**2).sum(axis=1)).min()
        if dmin - r <= (sigma_t + margin):
            keep[b] = pts
    return keep

def quick_viz_frame(mj_model, mj_data, local_clouds, obj_pts, contact_row, nearest_body=None, title="viz"):
    import trimesh
    rng = np.random.RandomState(0)
    cols = (rng.rand(52,3)*0.6+0.4)
    scene = trimesh.Scene()
    body_clouds = worldize_clouds_per_frame(mj_model, mj_data, local_clouds)
    for b, pts in body_clouds.items():
        if pts.size == 0: continue
        c = (cols[b]*255).astype(np.uint8); a = 255
        if contact_row[b]>0.5: c_vis = np.array([c[0],c[1],255,a], dtype=np.uint8)
        else:                  c_vis = np.array([int(0.6*c[0]), int(0.6*c[1]), int(0.6*c[2]), a], dtype=np.uint8)
        pc = trimesh.points.PointCloud(pts, colors=np.tile(c_vis,(pts.shape[0],1)))
        scene.add_geometry(pc, node_name=f"b{b:02d}")
    if nearest_body is None:
        pc2 = trimesh.points.PointCloud(obj_pts, colors=np.tile(np.array([20,200,20,255],np.uint8),(obj_pts.shape[0],1)))
    else:
        c = (cols[np.clip(nearest_body,0,51)]*255).astype(np.uint8)
        c = np.concatenate([c, np.full((c.shape[0],1),255,np.uint8)],1)
        pc2 = trimesh.points.PointCloud(obj_pts, colors=c)
    try:
        scene.show(title=title)
    except Exception as e:
        print("[viz] headless:", e)
