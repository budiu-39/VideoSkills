# -*- coding: utf-8 -*-

import mujoco
from collections import defaultdict
from scipy.spatial import cKDTree

import os
import numpy as np
import pickle
import trimesh

import torch
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


def set_scene_camera_facing_human(scene, root_pos):
    """
    设置 trimesh 场景相机，使其正对人体中心。
    root_pos: 人体根节点的 [x, y, z] 坐标
    """
    import trimesh
    import numpy as np

    # 1. 设定相机参数
    distance = 4.0  # 相机离人的距离
    elevation = 0.8  # 相机稍微抬高的高度 (Z轴)

    # 2. 计算相机位置 (从 Y 轴负方向看过来，通常是正面)
    cam_pos = root_pos + np.array([0, -distance, elevation])
    target = root_pos + np.array([0, 0, 0.4])  # 视点对准人的胸部高度

    # 3. 计算相机变换矩阵 (Look-at 矩阵)
    forward = target - cam_pos
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)

    up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward  # trimesh 相机朝向自己的 -Z
    mat[:3, 3] = cam_pos

    scene.camera_transform = mat

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


def quick_viz_ig_cg(
        body_local_clouds,
        obj_pts,
        body_pos_frame,  # 来自 sk_state.global_translation[t]
        body_rot_frame,  # 来自 sk_state.global_rotation[t]
        mj2sk,  # 必须传入：mj_body_id -> sk_index 的映射
        title="Corrected Viz",
        contact_row=None,  # (B,) 已按 Skeleton 顺序重排
        ig_frame=None,  # (B, 3) 已按 Skeleton 顺序重排
):
    import trimesh
    scene = trimesh.Scene()

    # 1. 预计算所有 Skeleton 节点的旋转矩阵
    # body_rot_frame 形状 (J, 4) -> (J, 3, 3)
    all_mats = sRot.from_quat(body_rot_frame).as_matrix()

    # 颜色定义
    RED, WHITE, BLUE, GREEN = [220, 40, 40, 255], [235, 235, 235, 255], [60, 60, 220, 255], [60, 200, 60, 120]

    # 2. 遍历每一个有采样点云的 MuJoCo Body
    for mj_bid, pts_local in body_local_clouds.items():
        # 获取该 Body 在 SkeletonState 数组中对应的索引
        if mj_bid not in mj2sk:
            continue
        sk_idx = mj2sk[mj_bid]

        # 严格对应位姿
        R = all_mats[sk_idx]
        T = body_pos_frame[sk_idx]

        # 变换点云到世界系: P_w = P_l * R^T + T
        pts_world = pts_local @ R.T + T

        # 判定颜色 (contact_row 是按 Skeleton 顺序排的)
        rgba = WHITE
        if contact_row is not None:
            val = contact_row[sk_idx]
            if val > 0.5:
                rgba = RED
            elif val < -0.5:
                rgba = BLUE

        pc = trimesh.points.PointCloud(pts_world, colors=np.tile(rgba, (pts_world.shape[0], 1)))
        scene.add_geometry(pc)

    # 3. 绘制物体 (绿色)
    obj_pc = trimesh.points.PointCloud(obj_pts, colors=np.tile(GREEN, (obj_pts.shape[0], 1)))
    scene.add_geometry(obj_pc)

    # 4. 绘制 IG 向量 (修正方向：从关节指向物体)
    if ig_frame is not None:
        starts = body_pos_frame

        # --- 核心修复点 ---
        # 如果你的 ig_mj 是 (物体最近点 - 关节中心)，则终点应该是 关节 + 向量
        ends = body_pos_frame + ig_frame  # 将 '-' 改为 '+'
        # -----------------

        valid = np.linalg.norm(ig_frame, axis=1) > 0.001
        if np.any(valid):
            segs = np.stack([starts[valid], ends[valid]], axis=1)
            path = trimesh.load_path(segs)
            for e in path.entities:
                e.color = [255, 140, 0, 255]  # 橙色
            scene.add_geometry(path)

    # 5. 地面与坐标轴
    scene.add_geometry(trimesh.creation.axis(axis_length=0.5))
    ground = trimesh.creation.box(extents=[5, 5, 0.005])
    ground.apply_translation([0, 0, -0.0025])
    ground.visual.face_colors = [200, 200, 200, 100]
    scene.add_geometry(ground)

    root_center = body_pos_frame[0]
    set_scene_camera_facing_human(scene, root_center)

    scene.show(title=title)

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
# ---------- 原语 SDF（几何局部坐标系） ----------
def sdf_sphere(p, r):                     return np.linalg.norm(p, axis=-1) - r
def sdf_box(p, half):
    q = np.abs(p) - half
    outside = np.maximum(q, 0.0)
    inside  = np.minimum(np.maximum(q[...,0], np.maximum(q[...,1], q[...,2])), 0.0)
    return np.linalg.norm(outside, axis=-1) + inside
def sdf_capsule_z(p, r, half):
    h = np.clip(p[...,2], -half, half)
    return np.sqrt(p[...,0]**2 + p[...,1]**2 + (p[...,2]-h)**2) - r
def sdf_cylinder_z(p, r, half):
    dx = np.linalg.norm(p[...,:2], axis=-1) - r
    dz = np.abs(p[...,2]) - half
    q = np.stack([dx, dz], axis=-1)
    outside = np.maximum(q, 0.0)
    return np.minimum(np.maximum(q[...,0], q[...,1]), 0.0) + np.linalg.norm(outside, axis=-1)


# ---------- world -> geomLocal（用 IG 的 body 位姿） ----------
def world_to_geom_local_ig(points_world, Rbw, pbw, geom_pos_body, geom_quat_wxyz):
    # geom 局部 -> body 局部
    q_wxyz = geom_quat_wxyz
    q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
    Rgb = sRot.from_quat(q_xyzw).as_matrix()  # (3,3)
    pgb = geom_pos_body                        # (3,)

    # geom 世界位姿
    Rgw = Rbw @ Rgb
    pgw = Rbw @ pgb + pbw

    # world -> geomLocal
    return (points_world - pgw) @ Rgw


def penetration_depth_sequence_ig(mj_model, body_geoms, mj2sk, obj_pts_seq, body_pos, body_rot):
    """
    计算物体点云与机器人各个 body 几何体之间的穿模深度。

    参数:
        mj_model: mujoco.MjModel
        body_geoms: dict, 由 build_local_templates_by_body 生成，包含 geom_id, type, size 等
        mj2sk: list/array, mj_body_id -> sk_joint_id 的映射
        obj_pts_seq: np.ndarray (T, P, 3) 或 List, 世界系物体点云
        body_pos: np.ndarray (T, B, 3), 骨骼全局位置 (SkeletonState.global_translation)
        body_rot: np.ndarray (T, B, 4), 骨骼全局旋转 (xyzw, SkeletonState.global_rotation)

    返回:
        pen_depth: np.ndarray (T, B), 每个关节在每一帧的最大穿模深度 (正值表示穿模)
    """
    T = len(obj_pts_seq)
    B = body_pos.shape[1]
    pen_depth = np.zeros((T, B), dtype=np.float32)

    for t in range(T):
        pts_w = obj_pts_seq[t]  # (P, 3)
        if len(pts_w) == 0: continue

        # 遍历所有几何体
        for mj_body_id, geoms in body_geoms.items():
            sk_idx = mj2sk.get(mj_body_id, -1)  # 使用 get 防止 key 缺失
            if sk_idx < 0: continue  # 跳过没有对应骨骼的 body (如 worldbody)

            # 获取当前 body 的世界系位姿
            b_pos = body_pos[t, sk_idx]
            b_rot = sRot.from_quat(body_rot[t, sk_idx])

            max_pen_for_body = 0.0

            for g_id in geoms:
                g_type = mj_model.geom_type[g_id]
                g_size = mj_model.geom_size[g_id]

                # 获取 geom 相对于 body 的局部偏移 (如果有的话)
                g_pos_local = mj_model.geom_pos[g_id]
                g_quat_local = mj_model.geom_quat[g_id]  # wxyz in mujoco
                g_rot_local = sRot.from_quat([g_quat_local[1], g_quat_local[2], g_quat_local[3], g_quat_local[0]])

                # 计算 geom 的世界系位姿
                # P_g_w = R_b_w * P_g_l + P_b_w
                geom_pos_w = b_rot.apply(g_pos_local) + b_pos
                geom_rot_w = b_rot * g_rot_local

                # 将物体点云转换到 geom 的局部坐标系
                # P_l = R_g_w^T * (P_w - P_g_w)
                pts_local = geom_rot_w.inv().apply(pts_w - geom_pos_w)

                # 根据几何体类型计算穿模
                if g_type == 3:  # mjtGeom.mjGEOM_CAPSULE
                    # Capsule 在 Mujoco 中沿 Z 轴生长, size = [radius, half_height, 0]
                    r, hh = g_size[0], g_size[1]
                    # 计算点到 Z 轴线段 [-hh, hh] 的距离
                    z_proj = np.clip(pts_local[:, 2], -hh, hh)
                    dist_to_axis = np.linalg.norm(pts_local[:, :2], axis=1)
                    dist_to_capsule = np.sqrt(dist_to_axis ** 2 + (pts_local[:, 2] - z_proj) ** 2) - r
                    # 穿模深度 = -dist (如果是负数表示在内部)
                    current_pen = -dist_to_capsule

                elif g_type == 2:  # mjtGeom.mjGEOM_BOX
                    # Box size = [half_x, half_y, half_z]
                    # 计算点到 AABB 的 Signed Distance
                    d = np.abs(pts_local) - g_size
                    # 简化的 SDF: 内部点的距离为负
                    dist_to_box = np.max(d, axis=1)
                    current_pen = -dist_to_box

                elif g_type == 5:  # mjtGeom.mjGEOM_CYLINDER
                    r, hh = g_size[0], g_size[1]
                    dist_r = np.linalg.norm(pts_local[:, :2], axis=1) - r
                    dist_z = np.abs(pts_local[:, 2]) - hh
                    current_pen = -np.maximum(dist_r, dist_z)

                else:
                    continue  # 暂不支持其他类型 (Sphere, Mesh等)

                if current_pen.max() > max_pen_for_body:
                    max_pen_for_body = current_pen.max()

            pen_depth[t, sk_idx] = max(0.0, max_pen_for_body)

    return pen_depth


def quick_viz_penetration(
        mj_model, body_local_clouds, obj_pts, body_pos_frame, body_rot_frame,
        mj2sk, pen_depth_row=None, title="Penetration Check (Trimesh)"
):
    """
    使用 trimesh 可视化单帧穿模情况
    """
    import trimesh
    scene = trimesh.Scene()

    # 颜色定义 [R, G, B, A]
    COLOR_NORMAL = [200, 200, 200, 60]  # 灰色半透明 (正常)
    COLOR_PENETRATE = [255, 0, 0, 200]  # 红色不透明 (穿模)
    COLOR_OBJECT = [0, 255, 0, 255]  # 绿色 (物体)

    # 1. 渲染机器人 Body 点云
    for mj_body_id, local_cloud in body_local_clouds.items():
        sk_idx = mj2sk.get(mj_body_id, -1)
        if sk_idx < 0:
            continue

        # 坐标变换
        pos = body_pos_frame[sk_idx]
        rot_mat = sRot.from_quat(body_rot_frame[sk_idx]).as_matrix()
        world_cloud = local_cloud @ rot_mat.T + pos

        # 判定颜色：根据该关节的穿模深度
        color = COLOR_NORMAL
        if pen_depth_row is not None and pen_depth_row[sk_idx] > 0.005:  # 阈值 5mm
            color = COLOR_PENETRATE

        # 创建点云对象
        # 采样：如果点太多，trimesh 可能会卡，这里可以根据需要 [::n]
        body_pc = trimesh.points.PointCloud(world_cloud, colors=np.tile(color, (world_cloud.shape[0], 1)))
        scene.add_geometry(body_pc)

    # 2. 渲染物体点云
    obj_pc = trimesh.points.PointCloud(obj_pts, colors=np.tile(COLOR_OBJECT, (obj_pts.shape[0], 1)))
    scene.add_geometry(obj_pc)

    # 3. 添加辅助装饰
    # 坐标轴
    scene.add_geometry(trimesh.creation.axis(axis_length=0.5))

    # 地面 (灰色平面)
    ground = trimesh.creation.box(extents=[10, 10, 0.001])
    ground.visual.face_colors = [150, 150, 150, 50]
    scene.add_geometry(ground)

    print(f"Showing scene: {title}")
    print("Controls: Left-Click to Rotate, Right-Click to Pan, Scroll to Zoom.")
    root_center = body_pos_frame[0]
    set_scene_camera_facing_human(scene, root_center)

    # 显示场景
    scene.show(title=title)


def get_aligned_vhacd_hulls(mesh_obj, cache_path, resolution=300000, max_hulls=64, max_v_per_ch=64):
    """
    使用 trimesh 的接口执行 V-HACD，参数对齐 Isaac Gym。
    """

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"--- Running VHACD via Trimesh (Resolution: {resolution}) ---")

    # trimesh 会根据系统环境调用可用的 VHACD 后端
    # 参数名称需要对应 trimesh 内部的映射逻辑
    hulls = mesh_obj.convex_decomposition(
        resolution=resolution,  # 对应 vhacd_params.resolution
        maxConvexHulls=max_hulls,  # 对应 vhacd_params.max_convex_hulls
        maxNumVerticesPerCH=max_v_per_ch  # 对应 vhacd_params.max_num_vertices_per_ch
    )

    # 兼容性处理：trimesh 有时返回单个 mesh，有时返回列表
    if not isinstance(hulls, list):
        hulls = [hulls]

    # 缓存结果
    with open(cache_path, "wb") as f:
        pickle.dump(hulls, f)

    return hulls


def compute_hull_planes(hull):
    """提取凸包所有平面的法线 n 和 偏移 d (dot(n,p) + d = 0)"""
    normals = hull.face_normals
    # 每个面上的一个顶点
    p0 = hull.vertices[hull.faces[:, 0]]
    d = -np.sum(normals * p0, axis=1)
    return normals, d


def prepare_batch_body_cloud(body_local_clouds, mj2sk, B_len):
    """
    预处理：将分散在各个 body 的点云合并成一个大数组，并记录索引映射。
    返回:
      all_local_pts: (Total_N, 3) 所有关节的局部点
      body_indices: (Total_N,) 每个点属于哪个关节 (sk_idx)
    """
    list_pts = []
    list_idxs = []

    # 确保按照 sk_idx 的顺序或者能映射回去
    # 这里我们直接遍历字典，建立映射
    for mj_bid, pts in body_local_clouds.items():
        sk_idx = mj2sk.get(mj_bid, -1)
        if sk_idx < 0 or sk_idx >= B_len: continue

        list_pts.append(pts)
        list_idxs.append(np.full(len(pts), sk_idx, dtype=np.int64))

    if not list_pts:
        return None, None

    all_local_pts = np.concatenate(list_pts, axis=0)  # (N, 3)
    body_indices = np.concatenate(list_idxs, axis=0)  # (N,)

    return all_local_pts, body_indices


def penetration_depth_vhacd_frame(
        body_pos_frame, body_rot_frame,
        all_body_local_pts, body_indices,
        hulls_planes_local, obj_pos, obj_rot_quat,
        obj_aabb_min, obj_aabb_max
):
    """
    极速版穿模计算：
    1. 批处理所有身体点
    2. 变换到物体局部坐标系 (Object Space)
    3. AABB 粗筛
    4. 仅对 AABB 内的点进行 V-HACD 细查
    """
    B = body_pos_frame.shape[0]
    pen_depths = np.zeros(B, dtype=np.float32)

    # --- 1. 将所有身体点变换到世界坐标系 (World Space) ---

    total_pts = len(all_body_local_pts)
    world_pts = np.empty((total_pts, 3), dtype=np.float32)

    unique_ids = np.unique(body_indices)

    # 预计算所有关节的旋转矩阵
    all_rots = sRot.from_quat(body_rot_frame).as_matrix()  # (B, 3, 3)

    for sk_idx in unique_ids:
        mask = (body_indices == sk_idx)
        # P_w = P_l @ R_body.T + T_body
        local_pts = all_body_local_pts[mask]
        R = all_rots[sk_idx]
        T = body_pos_frame[sk_idx]
        world_pts[mask] = local_pts @ R.T + T

    # --- 2. 将世界系点变换到物体局部系 (Object Local Space) ---
    # P_obj = (P_world - T_obj) @ R_obj
    # 这里的 rot 是从 Local 到 World，所以转回去要用 transpose (即 inverse)
    R_obj = sRot.from_quat(obj_rot_quat).as_matrix()  # (3, 3)
    # world_pts - obj_pos: (N, 3)
    # @ R_obj: (N, 3) @ (3, 3) -> (N, 3)
    pts_in_obj = (world_pts - obj_pos) @ R_obj

    # --- 3. Broad Phase: AABB 粗筛 ---
    # 检查哪些点在物体的 AABB 内。如果在外面，绝对不可能穿模。
    # obj_aabb_min/max 是物体在局部系下的包围盒
    # 稍微给 AABB 加一点点 padding (e.g. 1mm) 防止边界误差
    padding = 0.001
    in_aabb_mask = (
            (pts_in_obj[:, 0] >= obj_aabb_min[0] - padding) & (pts_in_obj[:, 0] <= obj_aabb_max[0] + padding) &
            (pts_in_obj[:, 1] >= obj_aabb_min[1] - padding) & (pts_in_obj[:, 1] <= obj_aabb_max[1] + padding) &
            (pts_in_obj[:, 2] >= obj_aabb_min[2] - padding) & (pts_in_obj[:, 2] <= obj_aabb_max[2] + padding)
    )

    if not np.any(in_aabb_mask):
        return pen_depths  # 没有任何点在物体附近

    # 仅保留通过粗筛的点
    candidate_pts = pts_in_obj[in_aabb_mask]
    candidate_indices = body_indices[in_aabb_mask]

    # --- 4. Narrow Phase: V-HACD 细查 ---
    # 计算这些点在每个凸包内的穿模深度
    # 初始化每个候选点的最大穿模深度为 0
    point_max_pen = np.zeros(len(candidate_pts), dtype=np.float32)

    for (normals, d) in hulls_planes_local:
        # normals: (F, 3), d: (F,)
        # dists: (N_cand, F)
        dists = candidate_pts @ normals.T + d

        # 判定是否在凸包内：所有面的距离都 < 0
        is_inside_hull = np.all(dists < 0, axis=1)  # (N_cand,)

        if np.any(is_inside_hull):
            # 穿模深度 = -max(dists) (离表面最近的距离)
            # update point_max_pen
            current_depths = -np.max(dists[is_inside_hull], axis=1)
            # 取最大值（因为一个点可能被多个重叠的凸包覆盖，虽少见但可能）
            point_max_pen[is_inside_hull] = np.maximum(point_max_pen[is_inside_hull], current_depths)

    # --- 5. 聚合结果到关节 ---
    # 只有 point_max_pen > 0 的才是真穿模
    penetrated_mask = point_max_pen > 0
    if np.any(penetrated_mask):
        pen_vals = point_max_pen[penetrated_mask]
        pen_idxs = candidate_indices[penetrated_mask]

        # 使用 reduceat 或者简单的循环取最大值
        # 这里用循环比较简单，因为穿模的关节通常不多
        unique_pen_joints = np.unique(pen_idxs)
        for sk_idx in unique_pen_joints:
            # 找到属于该关节的所有穿模点的深度，取最大
            max_d = np.max(pen_vals[pen_idxs == sk_idx])
            pen_depths[sk_idx] = max_d

    return pen_depths

def set_scene_camera_facing_human(scene, root_pos):
    """设置相机位置：Z-up系下正对人体"""
    distance = 4.0
    elevation = 1.0
    # 相机位于 Y 负半轴看向坐标原点（正面）
    cam_pos = root_pos + np.array([0, -distance, elevation])
    target = root_pos + np.array([0, 0, 0.5])

    forward = target - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    mat = np.eye(4)
    mat[:3, 0] = right
    mat[:3, 1] = up
    mat[:3, 2] = -forward
    mat[:3, 3] = cam_pos
    scene.camera_transform = mat


def quick_viz_penetration_vhacd(
        body_local_clouds, obj_pts, body_pos_frame, body_rot_frame,
        mj2sk, pen_depth_row=None, hulls_world=None, title="VHACD Penetration"
):
    """
    使用 trimesh 渲染。包含凸包线框显示。
    """
    import trimesh
    scene = trimesh.Scene()

    # 1. 绘制机器人
    for mj_bid, local_pts in body_local_clouds.items():
        sk_idx = mj2sk.get(mj_bid, -1)
        if sk_idx < 0: continue
        rot = sRot.from_quat(body_rot_frame[sk_idx]).as_matrix()
        world_pts = local_pts @ rot.T + body_pos_frame[sk_idx] # 注意变量名

        color = [180, 180, 180, 60]  # 正常灰色透明
        if pen_depth_row is not None and pen_depth_row[sk_idx] > 0.005:
            color = [255, 0, 0, 180]  # 穿模红色

        # 修复：将 world_cloud 改为 world_pts
        scene.add_geometry(trimesh.points.PointCloud(world_pts, colors=np.tile(color, (world_pts.shape[0], 1))))

    # 2. 绘制物体点云
    scene.add_geometry(trimesh.points.PointCloud(obj_pts, colors=np.tile([0, 255, 0, 255], (obj_pts.shape[0], 1))))

    # 3. 绘制物体凸包线框
    if hulls_world:
        for h in hulls_world:
            edge = h.copy()
            edge.visual.face_colors = [0, 255, 0, 30] # 浅绿色凸包
            scene.add_geometry(edge)

    # 4. 相机正对人
    set_scene_camera_facing_human(scene, body_pos_frame[0])
    scene.show(title=title)


def quat_to_rotmat_torch(quat):
    """
    PyTorch 版四元数转旋转矩阵。
    quat: (..., 4) [x, y, z, w] (SciPy 格式)
    返回: (..., 3, 3)
    """
    # 归一化
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # 构造矩阵
    out = torch.empty(quat.shape[:-1] + (3, 3), device=quat.device, dtype=quat.dtype)

    out[..., 0, 0] = 1 - 2 * (y2 + z2)
    out[..., 0, 1] = 2 * (xy - wz)
    out[..., 0, 2] = 2 * (xz + wy)

    out[..., 1, 0] = 2 * (xy + wz)
    out[..., 1, 1] = 1 - 2 * (x2 + z2)
    out[..., 1, 2] = 2 * (yz - wx)

    out[..., 2, 0] = 2 * (xz - wy)
    out[..., 2, 1] = 2 * (yz + wx)
    out[..., 2, 2] = 1 - 2 * (x2 + y2)

    return out


def penetration_depth_vhacd_cuda(
        body_pos_seq, body_rot_seq,  # (T, B, 3), (T, B, 4)
        all_body_local_pts, body_indices,  # (N, 3), (N,)
        hulls_planes_local,  # List[(F, 3), (F,)]
        obj_pos_seq, obj_rot_seq,  # (T, 3), (T, 4)
        obj_aabb_min, obj_aabb_max,  # (3,)
        device='cuda',
        chunk_size=100  # <--- 新增：每次处理的帧数，默认100，显存小可调小
):
    """
    全序列 CUDA 并行穿模计算（分块版，防止 OOM）。
    """

    # --- 1. 数据类型与设备转换 ---
    # 使用 .clone().detach() 或 np.copy() 解决 "non-writable" 警告
    def to_tensor(x, dtype):
        if isinstance(x, np.ndarray):
            # 解决 numpy read-only 警告
            x = torch.from_numpy(np.copy(x))
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return x.to(device=device, dtype=dtype)

    body_pos_seq = to_tensor(body_pos_seq, torch.float32)
    body_rot_seq = to_tensor(body_rot_seq, torch.float32)
    all_body_local_pts = to_tensor(all_body_local_pts, torch.float32)
    body_indices = to_tensor(body_indices, torch.long)  # 索引必须是 long
    obj_pos_seq = to_tensor(obj_pos_seq, torch.float32)
    obj_rot_seq = to_tensor(obj_rot_seq, torch.float32)
    obj_aabb_min = to_tensor(obj_aabb_min, torch.float32)
    obj_aabb_max = to_tensor(obj_aabb_max, torch.float32)

    # 预处理平面方程 (list of tensors)
    planes_cuda = []
    for n, d in hulls_planes_local:
        planes_cuda.append((
            to_tensor(n, torch.float32),
            to_tensor(d, torch.float32)
        ))

    T_total, B, _ = body_pos_seq.shape
    N_pts = all_body_local_pts.shape[0]

    # 结果列表
    pen_seq_chunks = []

    # --- 2. 分块循环 (Chunk Loop) ---
    # 将 T 维度切分为多个小 batch，防止显存爆炸
    for start_t in range(0, T_total, chunk_size):
        end_t = min(start_t + chunk_size, T_total)

        # 切片数据
        b_pos_sub = body_pos_seq[start_t:end_t]
        b_rot_sub = body_rot_seq[start_t:end_t]
        o_pos_sub = obj_pos_seq[start_t:end_t]
        o_rot_sub = obj_rot_seq[start_t:end_t]

        t_sub = end_t - start_t

        # === 以下是原来的计算逻辑 (针对当前 chunk) ===

        # 1. 扩展索引
        p_indices = body_indices.view(1, -1).expand(t_sub, -1)  # (t_sub, N_pts)

        # 2. Gather Body Pose
        idx_pos = p_indices.unsqueeze(-1).expand(-1, -1, 3)
        batch_pts_pos = torch.gather(b_pos_sub, 1, idx_pos)

        idx_rot = p_indices.unsqueeze(-1).expand(-1, -1, 4)
        batch_pts_quat = torch.gather(b_rot_sub, 1, idx_rot)

        # 3. Local Point -> World Point
        batch_rot_mat = quat_to_rotmat_torch(batch_pts_quat)  # (t, N, 3, 3)
        pts_local_exp = all_body_local_pts.view(1, N_pts, 3, 1)
        pts_world = (batch_rot_mat @ pts_local_exp).squeeze(-1) + batch_pts_pos

        # 4. World Point -> Object Local Point
        R_obj = quat_to_rotmat_torch(o_rot_sub)  # (t, 3, 3)
        pts_rel = pts_world - o_pos_sub.unsqueeze(1)
        # R^T @ P = (P @ R) if row-major? PyTorch matmul last 2 dims.
        # R_obj is Obj->World. We need World->Obj (R^T).
        # matmul( (t,3,3).T, (t,N,3,1) ) -> (t,3,3) need unsqueeze to (t,1,3,3)
        # 简便写法: P_local = (P_world - T) @ R_obj
        # 验证: vec(1,3) @ mat(3,3) = vec(1,3).
        pts_in_obj = torch.matmul(pts_rel, R_obj)

        # 5. AABB Broad Phase
        padding = 0.005
        in_aabb = (
                (pts_in_obj[..., 0] >= obj_aabb_min[0] - padding) & (pts_in_obj[..., 0] <= obj_aabb_max[0] + padding) &
                (pts_in_obj[..., 1] >= obj_aabb_min[1] - padding) & (pts_in_obj[..., 1] <= obj_aabb_max[1] + padding) &
                (pts_in_obj[..., 2] >= obj_aabb_min[2] - padding) & (pts_in_obj[..., 2] <= obj_aabb_max[2] + padding)
        )

        # 当前 Chunk 的结果容器
        pen_sub = torch.zeros((t_sub, B), device=device, dtype=torch.float32)

        # 如果有任何点在 AABB 内，才进行精细检测
        if in_aabb.any():
            active_mask = in_aabb
            active_pts = pts_in_obj[active_mask]  # Flattened (K, 3)

            # 记录哪些点属于哪个 (t, b)
            grid_t = torch.arange(t_sub, device=device).view(-1, 1).expand(-1, N_pts)
            grid_b = body_indices.view(1, -1).expand(t_sub, -1)

            active_t = grid_t[active_mask]
            active_b = grid_b[active_mask]

            active_pen = torch.zeros(active_pts.shape[0], device=device, dtype=torch.float32)

            # V-HACD Narrow Phase
            for (n_cuda, d_cuda) in planes_cuda:
                # n: (F, 3), d: (F,)
                dists = active_pts @ n_cuda.T + d_cuda
                is_inside = torch.all(dists < 0, dim=1)

                if is_inside.any():
                    # depth = -max(dist)
                    depths = -torch.max(dists[is_inside], dim=1)[0]
                    # Update max penetration
                    current = active_pen[is_inside]
                    active_pen[is_inside] = torch.maximum(current, depths)

            # Scatter Reduce (Max pooling to joints)
            flat_idx = active_t * B + active_b
            pen_flat = torch.zeros(t_sub * B, device=device, dtype=torch.float32)
            pen_flat.scatter_reduce_(0, flat_idx, active_pen, reduce='amax', include_self=False)
            pen_sub = pen_flat.view(t_sub, B)

        pen_seq_chunks.append(pen_sub)

        # 可选：清理缓存
        # torch.cuda.empty_cache()

    # 合并结果
    return torch.cat(pen_seq_chunks, dim=0)