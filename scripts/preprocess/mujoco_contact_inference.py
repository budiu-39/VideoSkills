# -*- coding: utf-8 -*-
import numpy as np
import mujoco
import trimesh
from collections import defaultdict
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as sRot

# ===== 常量（与你现有阈值保持一致，可按需调整） =====
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)   # Z-up
AIRBORNE_H = 0.05         # 空中判定高度阈值（m）
ACC_DEV_THR = 2.0         # 偏离重力阈值（m/s^2）
GROUND_H_TOL = 0.02       # 认为在地面的 z 容差（m）
SPEED_THR = 0.05          # 地面仍在运动的速度阈值（m/s）
CLOSE_MIN_THR = 0.05      # 全局最近点阈值（m）
ADAPTIVE_PAD = 0.005      # sigma_t = min_dist + 0.005
FIXED_SIGMA_NO_INTERACT = 0.02  # 不满足交互条件的保底阈值


# ===== 1) 从 XML 采样 local 模板，按 body 聚合（只支持新版 dict） =====
import numpy as np
import mujoco
import trimesh
from collections import defaultdict
from scipy.spatial.transform import Rotation as sRot



def build_local_templates_by_body(xml_path, samples_per_geom=1500, drop_world=True):
    """
    读取 MuJoCo XML，按每个 geom 采样表面点，并先用 (geom_quat, geom_pos) 把它们
    从“geom 局部”变到“body 局部”，再按 body 聚合。
    返回:
      body_local_clouds: Dict[int, (Ni,3)]  # body 局部坐标
      body_geoms:        Dict[int, List[int]]
      mj_model:          mujoco.MjModel
    """
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # 预取 mesh 几何
    mesh_trimesh = []
    for m in range(mj_model.nmesh):
        adr, n = mj_model.mesh_vertadr[m], mj_model.mesh_vertnum[m]
        verts = mj_model.mesh_vert[adr:adr+n].copy()
        fadr, fn = mj_model.mesh_faceadr[m], mj_model.mesh_facenum[m]
        faces = mj_model.mesh_face[fadr:fadr+fn].copy()
        mesh_trimesh.append(trimesh.Trimesh(vertices=verts, faces=faces, process=False))

    # 需要的 model 数组
    geom2body = mj_model.geom_bodyid.copy()
    geom_type  = mj_model.geom_type.copy()
    geom_size  = mj_model.geom_size.copy()
    geom_pos   = mj_model.geom_pos.copy()            # (ngeom,3)
    geom_quat  = mj_model.geom_quat.copy()           # (ngeom,4)  注意: MuJoCo 为 **wxyz**

    agg = defaultdict(list)
    body_geoms = defaultdict(list)

    for g in range(mj_model.ngeom):
        gtype = geom_type[g]
        size  = geom_size[g]
        pts_g = None  # 在 geom 局部坐标系下的点

        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(size[0])
            tm = trimesh.creation.icosphere(subdivisions=3, radius=r)
            pts_g, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = size[:3].astype(float)
            tm = trimesh.creation.box(extents=half * 2.0)
            pts_g, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r, half = float(size[0]), float(size[1])
            tm = trimesh.creation.capsule(radius=r, height=2.0*half, count=[32, 8])  # 轴向=Z
            pts_g, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, half = float(size[0]), float(size[1])
            tm = trimesh.creation.cylinder(radius=r, height=2.0*half, sections=48)   # 轴向=Z
            pts_g, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mid = mj_model.geom_dataid[g]
            tm = mesh_trimesh[mid]
            pts_g, _ = trimesh.sample.sample_surface_even(tm, count=samples_per_geom)

        else:
            # plane/heightfield/ellipsoid 等跳过
            continue

        b = int(geom2body[g])
        if drop_world and b == 0:
            continue
        if pts_g is None or pts_g.size == 0:
            continue

        # ===== 关键：geom 局部 -> body 局部 =====
        q_wxyz = geom_quat[g]                                  # MuJoCo 存储为 wxyz
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)  # SciPy 期望 xyzw
        Rgb = sRot.from_quat(q_xyzw).as_matrix()               # (3,3)
        pgb = geom_pos[g]                                      # (3,)
        pts_body = pts_g @ Rgb.T + pgb                         # (K,3) —— 现在在 body 局部系

        agg[b].append(pts_body.astype(np.float32))
        body_geoms[b].append(g)

    body_local_clouds = {b: (np.concatenate(v, axis=0) if len(v) > 1 else v[0])
                         for b, v in agg.items()}
    return body_local_clouds, dict(body_geoms), mj_model



# ===== 2) 将按 body 的局部点云，变换到世界坐标 =====
def worldize_clouds_per_frame(mj_data, body_local_clouds):
    out = {}
    for b, pts_local in body_local_clouds.items():
        if pts_local is None or pts_local.size == 0:
            continue
        b = int(b)
        Rbw = mj_data.xmat[b].reshape(3, 3)  # (9,) -> (3,3)
        pbw = mj_data.xpos[b]                # (3,)
        out[b] = pts_local @ Rbw.T + pbw
    return out

def worldize_clouds_per_frame_sk(ig_body_pos_world, ig_body_rot_world, body_local_clouds, mj2sk):
    out = {}
    for b, pts_local in body_local_clouds.items():
        if pts_local is None or pts_local.size == 0:
            continue
        i = int(mj2sk[b])
        Rbw = sRot.from_quat(ig_body_rot_world).as_matrix()[i]  # (9,) -> (3,3)
        pbw = ig_body_pos_world[i]             # (3,)
        out[b] = pts_local @ Rbw.T + pbw
    return out


# ===== 3) KDTree 最近邻：得到每个物体点最近 body，及各 body 的最小距离 =====
def min_dist_per_body_body2obj_single_tree(obj_pts, body_world_clouds, n_bodies, workers=1):
    """
    方向：body 点 -> 物体点
    目标：每个 body 到物体表面的最小距离（严格意义上的 per-body 最近距离）
    输入：
      obj_pts: (P,3)  物体表面点（世界系）
      body_world_clouds: Dict[int] -> (Mi,3)  每个 body 的点云（世界系）
      n_bodies: int
    返回：
      min_row: (n_bodies,)  每个 body 的最近距离（没点或缺席者为 inf）
      d_all: float           全局最小距离 = min(min_row)
    """
    if obj_pts is None or len(obj_pts) == 0:
        return np.full((n_bodies,), np.inf, np.float32), np.inf

    tree_obj = cKDTree(obj_pts)  # 物体点建树（一次）
    min_row = np.full((n_bodies,), np.inf, np.float32)

    for b, pts in body_world_clouds.items():
        if pts is None or pts.size == 0:
            continue
        # 对 body 的每个点，查“最近物体点”的距离
        dmin_b, _ = tree_obj.query(pts, k=1, workers=workers)  # (Mi,)
        md = float(np.min(dmin_b))
        min_row[int(b)] = md

    d_all = float(np.min(min_row)) if np.isfinite(min_row).any() else np.inf
    return min_row, d_all


# ===== 4) 接触计算主入口（只支持新版 dict） =====
def contacts_from_xml_pointcloud(
    mj_model, body_local_clouds,
    qpos_seq,                  # (T, nq)
    obj_pts_world,             # (T, P, 3)
    obj_pos, vel, acc,         # (T, 3)
    sigma_pad=ADAPTIVE_PAD, sigma_no_interact=FIXED_SIGMA_NO_INTERACT,
    ground_height=0.0,
    show_progress=True, progress_desc="contacts",
    workers=1,
    sk2mj=None, mj2sk=None,     # Skeleton 顺序 -> MuJoCo body id 的映射
    ig_body_pos_world=None,
    ig_body_rot_world=None,
    ground_mask_sk=None
):
    """
    输出:
      contact: (T, B)     # B = len(sk2mj)
      ig:      (T, B, 3)  # ig[t,b] = body_xpos[t,b] - NN(obj_pts_world[t])
    """
    T = qpos_seq.shape[0]

    # ---- 映射：若未传，则用 body_local_clouds 的实际键集合（升序），不再假设从 0 连续 ----
    if sk2mj is None or mj2sk is None:
        keys = sorted(int(k) for k in body_local_clouds.keys())  # e.g. [1,2,...,52]
        sk2mj = keys

    B = len(sk2mj)
    contact = np.zeros((T, B), dtype=np.float32)
    ig      = np.zeros((T, B, 3), dtype=np.float32)

    mj_data = mujoco.MjData(mj_model)
    it = range(T)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc=progress_desc, leave=False)
        except Exception:
            pass

    for t in it:
        mj_data.qpos[:] = qpos_seq[t]
        mujoco.mj_forward(mj_model, mj_data)

        # —— 世界系点云（只取 sk2mj 覆盖的 body） ——
        if ig_body_pos_world is not None and ig_body_rot_world is not None:
            body_world_full = worldize_clouds_per_frame_sk(ig_body_pos_world[t], ig_body_rot_world[t], body_local_clouds, mj2sk)
        else:
            body_world_full = worldize_clouds_per_frame(mj_data, body_local_clouds)

        body_world = {b: body_world_full[b] for b in sk2mj if b in body_world_full and body_world_full[b].size > 0}
        if not body_world or obj_pts_world[t].size == 0:
            continue

        # —— ig 用 MJCF body 原点（按 Skeleton 顺序） ——
        if ig_body_pos_world is not None:
            # 这里假设 ig_body_pos_world 的第二维已经与 sk2mj 对齐（= Skeleton 顺序）
            body_pos_for_ig = ig_body_pos_world[t].astype(np.float32)  # (B,3)
            assert body_pos_for_ig.shape[0] == B, "ig_body_pos_world 与 sk2mj 长度不一致"
        else:
            # 回退：用 MJCF body 原点（旧做法）
            body_pos_for_ig = mj_data.xpos[sk2mj].copy().astype(np.float32)

        tree = cKDTree(obj_pts_world[t])
        d, idx = tree.query(body_pos_for_ig, k=1, workers=workers)       # (B,)
        nn = obj_pts_world[t][idx]                                  # (B,3)
        ig[t] = body_pos_for_ig - nn                                     # (B,3)

        # —— 各 body 最小距离（对“几何点云”），n_bodies 用 mj_model.nbody 作为全局索引空间 ——
        # 研究下这里
        min_row, d_all = min_dist_per_body_body2obj_single_tree(
            obj_pts_world[t], body_world, n_bodies=mj_model.nbody, workers=workers
        )

        # 直接用映射选出 Skeleton 顺序的行（未参与的 body 会是 inf）
        min_row_sk = min_row[sk2mj]  # (B,)

        # —— 自适应阈值判定 contact（原逻辑） ——
        # GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float32)   # Z-up
        # AIRBORNE_H = 0.05         # 空中判定高度阈值（m）
        # ACC_DEV_THR = 2.0         # 偏离重力阈值（m/s^2）
        # GROUND_H_TOL = 0.02       # 认为在地面的 z 容差（m）
        # SPEED_THR = 0.05          # 地面仍在运动的速度阈值（m/s）
        # CLOSE_MIN_THR = 0.02      # 全局最近点阈值（m）
        # ADAPTIVE_PAD = 0.005      # sigma_t = min_dist + 0.005
        # FIXED_SIGMA_NO_INTERACT = 0.02  # 不满足交互条件的保底阈值
        airborne = (obj_pos[t, 2] - ground_height) > AIRBORNE_H
        acc_dev = np.linalg.norm(acc[t] - GRAVITY) > ACC_DEV_THR
        near_g = abs(obj_pos[t, 2] - ground_height) <= GROUND_H_TOL
        moving = np.linalg.norm(vel[t]) > SPEED_THR
        interacted = (airborne and acc_dev) or (near_g and moving) or (d_all < CLOSE_MIN_THR)

        # 促成带阈值（红带）：参考 Sec.C.1
        sigma_pos = float(d_all + sigma_pad) if interacted else float(sigma_no_interact)  # ≈  min_dist+0.005  或 0.02

        # 惩罚带阈值（蓝带）：参考 Sec.D.2
        sigma_neg = 0.10  # 固定 10cm

        # 三值化：+1(促成) / 0(中性) / -1(惩罚)
        row = min_row_sk  # (B,)
        tristate = np.zeros_like(row, dtype=np.int8)

        # +1：进入促成带
        tristate[row < sigma_pos] = +1


        # -1：远离到惩罚带之外，且不是与地面的刚体（可选：脚底排除）
        # 这里 ground_mask 你可以按自己数据集来定；没有就先不排
        far_mask = row > sigma_neg
        if ground_mask_sk is not None:
            far_mask &= ~ground_mask_sk
        tristate[far_mask] = -1


        # 中性：剩下的自然是 0
        contact[t, :] = tristate  # 新增：-1/0/+1

    return contact, ig




# ===== 5) 可视化：只高亮 contact body=红，其余=白；加地面与坐标轴 =====
def quick_viz_frame(
    mj_model, mj_data, body_local_clouds,
    obj_pts, contact_row,
    body_pos_frame,
    body_rot_frame,
    mj2sk,
    title="viz", ground_size=10.0,
    ig_frame=None,
):
    scene = trimesh.Scene()

    # 世界系点云（按 body）
    body_world = worldize_clouds_per_frame_sk(body_pos_frame, body_rot_frame, body_local_clouds, mj2sk)

    # —— 颜色表 ——
    RED    = np.array([220,  40,  40, 255], np.uint8)   # +1: 促成
    WHITE  = np.array([235, 235, 235, 255], np.uint8)   #  0: 中性
    BLUE   = np.array([ 60,  60, 220, 255], np.uint8)   # -1: 惩罚
    GREEN  = np.array([ 60, 200,  60, 255], np.uint8)   # 物体点云
    CYAN   = np.array([ 50, 150, 255, 255], np.uint8)   # body_pos 标记

    # —— 机器人各 body：根据 contact 三值渲染 ——
    B = len(contact_row) if contact_row is not None else mj_model.nbody
    for b, pts in body_world.items():
        if pts is None or pts.size == 0:
            continue

        if contact_row is not None:
            val = float(contact_row[mj2sk[b]])
            if val > 0.5:      # +1 or binary 1
                rgba = RED
            elif val < -0.5:   # -1
                rgba = BLUE
            else:              # 0 or neutral
                rgba = WHITE
        else:
            rgba = WHITE

        pc = trimesh.points.PointCloud(pts, colors=np.tile(rgba, (pts.shape[0], 1)))
        scene.add_geometry(pc, node_name=f"body_{b:02d}")

    # —— 物体点云（绿色） ——
    obj_pc = trimesh.points.PointCloud(obj_pts, colors=np.tile(GREEN, (obj_pts.shape[0], 1)))
    scene.add_geometry(obj_pc, node_name="object_points")

    # —— 人体代表点（青色） ——
    if body_pos_frame is not None:
        bp = np.asarray(body_pos_frame, dtype=np.float32)
        bp_pc = trimesh.points.PointCloud(bp, colors=np.tile(CYAN, (bp.shape[0], 1)))
        scene.add_geometry(bp_pc, node_name="body_markers")

    # —— ig 线段（橙色） ——
    if (body_pos_frame is not None) and (ig_frame is not None):
        starts = np.asarray(body_pos_frame, dtype=np.float32)
        ends   = starts + np.asarray(ig_frame, dtype=np.float32)
        segs   = np.stack([starts, ends], axis=1)  # (B,2,3)
        path_orange = trimesh.load_path(segs)
        color_orange = np.tile(np.array([[255, 140, 0, 255]], dtype=np.uint8), (len(segs), 1))
        for i in range(len(path_orange.entities)):
            path_orange.entities[i].color = color_orange[i]
        scene.add_geometry(path_orange, node_name="ig_lines_body")

    # —— 地面 + 坐标轴 ——
    plane = trimesh.creation.box(extents=(ground_size, ground_size, 0.01))
    plane.apply_translation([0.0, 0.0, -0.005])
    plane.visual.face_colors = [200, 200, 200, 120]
    scene.add_geometry(plane, node_name="ground")
    scene.add_geometry(trimesh.creation.axis(axis_length=1.0), node_name="axes")

    try:
        scene.show(title=title)
    except Exception as e:
        print("[viz] headless:", e)

def build_qpos_seq_from_state(mj_model, new_sk_state):
    """
    生成 (T, nq) 的 qpos：
      qpos[:3]   = root_trans
      qpos[3:7]  = root_quat (wxyz)      # MuJoCo 约定
      qpos[7:]   = 其余关节轴角平铺（每关节 3 维）
    要求：
      - new_sk_state.global_root_rotation: (T,4) xyzw
      - new_sk_state.local_rotation[:,1:]: (T, J-1, 4) xyzw（SciPy约定）
    """
    T = new_sk_state.root_translation.shape[0]
    nq = mj_model.nq
    qpos_seq = np.zeros((T, nq), dtype=np.float32)

    # root（wxyz）
    root_trans = new_sk_state.root_translation.numpy().astype(np.float32)         # (T,3)
    root_quat_xyzw = new_sk_state.global_root_rotation.numpy().astype(np.float32) # (T,4) wxyz
    root_quat_wxyz = root_quat_xyzw[..., [3, 0, 1, 2]]
    qpos_seq[:, :3] = root_trans
    qpos_seq[:, 3:7] = root_quat_wxyz

    # 其余关节（xyzw -> 轴角）
    local_quat = new_sk_state.local_rotation[:, 1:].reshape(T, -1, 4).numpy()    # (T, J-1, 4) xyzw
    dof_axis_angle = sRot.from_quat(local_quat.reshape(-1, 4)).as_rotvec().reshape(T, -1)  # (T, 3*(J-1))

    assert qpos_seq.shape[1] == 7 + dof_axis_angle.shape[1], \
        f"nq mismatch: model.nq={qpos_seq.shape[1]}, dof_len={7 + dof_axis_angle.shape[1]}"
    qpos_seq[:, 7:] = dof_axis_angle.astype(np.float32)
    return qpos_seq


def build_sk2mj_index(mj_model, skeleton_tree, drop_world=True):
    """
    返回：
      sk2mj: [B]  skeleton_tree.node_names 的顺序在 MuJoCo 里的 body id
      mj2sk: dict{ mj_body_id -> sk_row }  反向映射
    要求：MJCF 里 body 的名字与 SkeletonTree.from_mjcf 解析的名字一致
    """
    # 先做一个 name->mj_id 的表
    mj_name2id = {}
    for bid in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if name is not None:
            mj_name2id[name] = bid

    # 按 SkeletonTree 的顺序取对应的 mj_id
    sk2mj = []
    for nm in skeleton_tree.node_names:
        if nm not in mj_name2id:
            raise KeyError(f"Body name '{nm}' not found in MJCF.")
        sk2mj.append(mj_name2id[nm])

    # 是否跳过 world：通常 Skeleton 的第一个就是 root（不是 world）
    # 如果你的 skeleton 里没有 world，这里不需要删
    if drop_world and len(sk2mj) > 0 and sk2mj[0] == 0:
        sk2mj = sk2mj[1:]

    mj2sk = {mj_id: i for i, mj_id in enumerate(sk2mj)}
    return sk2mj, mj2sk
