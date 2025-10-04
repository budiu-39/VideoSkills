# scripts/preprocess/hoi_visualization.py
import numpy as np
import trimesh

# 可选：你项目里已有 worldize_clouds_per_frame 可直接导入；
# 若不想跨文件依赖，可以把它粘过来。
try:
    from scripts.preprocess.mujoco_contact_inference import worldize_clouds_per_frame
except Exception:
    worldize_clouds_per_frame = None
    print("[hoi_visualization] WARNING: cannot import worldize_clouds_per_frame. "
          "Pass `body_clouds` manually or provide the function.")

def _color_table(n, seed=42):
    """生成 n 个较亮的随机颜色（RGB，0~255）"""
    rng = np.random.RandomState(seed)
    cols = (rng.rand(n, 3) * 0.6 + 0.4)  # 亮一点
    return (cols * 255).astype(np.uint8)

def _make_line_set(starts, ends, color_rgba=(255, 120, 0, 255)):
    """
    用线段可视化 ig：给定起点、终点，返回 trimesh Path3D
    starts/ends: [K,3]
    """
    segs = np.stack([starts, ends], axis=1)  # [K,2,3]
    path = trimesh.load_path(segs)
    # trimesh Path 没有逐段颜色接口，这里用统一颜色；需要多色的话拆多段
    return path

def _infer_num_bodies(body_clouds=None, contact_row=None, default_n=52):
    """
    推断 body 数量的策略：
    1) 若给了 contact_row，就用 len(contact_row)
    2) 否则用 body_clouds 的最大 key + 1
    3) 都没有时退回默认值
    """
    if contact_row is not None:
        return int(np.asarray(contact_row).shape[0])
    if body_clouds:
        try:
            mx = max(int(k) for k in body_clouds.keys())
            return mx + 1
        except Exception:
            pass
    return default_n

def _safe_body_clouds(body_clouds, num_bodies):
    """把不在 [0, num_bodies-1] 的 key 过滤掉，避免越界"""
    clean = {}
    for k, v in body_clouds.items():
        if isinstance(k, (int, np.integer)) and 0 <= int(k) < num_bodies:
            clean[int(k)] = v
    return clean

def compute_nearest_body_colors(obj_pts, body_clouds, cols, workers=1):
    """
    给物体点上色：按最近 body 着色。
    返回 colors: [P,4] (RGBA) 和 nearest_body: [P]
    """
    from scipy.spatial import cKDTree
    ALL, OWN = [], []
    for b, pts in body_clouds.items():
        if pts is None or pts.size == 0:
            continue
        ALL.append(pts)
        OWN.append(np.full((pts.shape[0],), int(b), np.int32))
    if len(ALL) == 0:
        return np.tile(np.array([20, 200, 20, 255], np.uint8), (obj_pts.shape[0], 1)), None
    ALL = np.vstack(ALL)
    OWN = np.concatenate(OWN)
    tree = cKDTree(ALL)
    dmin, idx = tree.query(obj_pts, k=1, workers=workers)
    nearest_body = OWN[idx]
    nb_clip = np.clip(nearest_body, 0, cols.shape[0]-1)
    rgb = cols[nb_clip]
    rgba = np.concatenate([rgb, np.full((rgb.shape[0], 1), 255, np.uint8)], axis=1)
    return rgba, nearest_body

def visualize_contact_frame(
    mj_model=None, mj_data=None, local_clouds=None,
    obj_pts_world_frame=None,      # [P,3]  该帧物体点（世界系）
    body_pos_frame=None,           # [B,3]  该帧身体关键点（世界系），可选
    ig_frame=None,                 # [B,3]  ig 向量（points1 - NN(points2)），可选
    contact_row=None,              # [B]    0/1 是否接触（按阈值），可选
    body_clouds=None,              # dict{ body_id -> [Mi,3] }，可选（若未提供且具备 mj_* 则自动生成）
    title="contact_debug",
    show=True,
    save_path=None,                # 如 "debug_frame.glb"/".ply"
    highlight_scale=1.0,           # 被命中 body 的亮度提升比例（可微调可视效果）
    workers_nn=1                   # 最近体着色时 KDTree 并行线程数
):
    """
    可视化一帧：
      - 机器人每个 body 的采样点云（不同颜色）
      - 物体点云（可按最近 body 着色）
      - 身体关键点（红色）
      - ig 向量（橙色线段，从 body 指向最近物体点：end = body - ig）
    """
    assert obj_pts_world_frame is not None and obj_pts_world_frame.ndim == 2 and obj_pts_world_frame.shape[1] == 3, \
        "obj_pts_world_frame must be [P,3] numpy array."
    scene = trimesh.Scene()

    # 1) 准备 body_clouds
    if body_clouds is None:
        if mj_model is None or mj_data is None or local_clouds is None:
            raise ValueError("Either provide (mj_model, mj_data, local_clouds) or body_clouds.")
        if worldize_clouds_per_frame is None:
            raise RuntimeError("worldize_clouds_per_frame not available. Import or pass body_clouds.")
        body_clouds = worldize_clouds_per_frame(mj_model, mj_data, local_clouds)

    # 2) 推断 body 数、生成颜色表、过滤越界 key
    num_bodies = _infer_num_bodies(body_clouds, contact_row, default_n=52)
    cols = _color_table(num_bodies)
    body_clouds = _safe_body_clouds(body_clouds, num_bodies)

    # 3) 机器人点云
    for b, pts in body_clouds.items():
        if pts is None or pts.size == 0:
            continue
        base = cols[b].astype(np.float32)
        if contact_row is not None:
            on = (b < len(contact_row)) and (float(contact_row[b]) > 0.5)
        else:
            on = False
        if on:
            rgb = np.clip(base * max(1.0, 1.2 * highlight_scale), 0, 255).astype(np.uint8)
            # 高亮：蓝通道拉满
            rgb = np.array([rgb[0], rgb[1], 255], dtype=np.uint8)
        else:
            rgb = (0.6 * base).astype(np.uint8)
        rgba = np.concatenate([rgb, np.array([255], np.uint8)], axis=0)
        pc = trimesh.points.PointCloud(pts, colors=np.tile(rgba, (pts.shape[0], 1)))
        scene.add_geometry(pc, node_name=f"body_{b}")

    # 4) 物体点云
    #    若提供最近 body（nearest_body），物体点可按最近 body 着色；否则用统一绿色
    try:
        rgba, nearest_body = compute_nearest_body_colors(obj_pts_world_frame, body_clouds, cols, workers=workers_nn)
        obj_pc = trimesh.points.PointCloud(obj_pts_world_frame, colors=rgba)
    except Exception as e:
        # 退化为统一颜色
        obj_pc = trimesh.points.PointCloud(
            obj_pts_world_frame,
            colors=np.tile(np.array([20, 200, 20, 255], np.uint8), (obj_pts_world_frame.shape[0], 1))
        )
    scene.add_geometry(obj_pc, node_name="object_points")

    # 5) 身体关键点（可选）
    if body_pos_frame is not None:
        bp = np.asarray(body_pos_frame)
        bp_col = np.tile(np.array([255, 50, 50, 255], np.uint8), (bp.shape[0], 1))
        scene.add_geometry(trimesh.points.PointCloud(bp, colors=bp_col), node_name="body_markers")

    # 6) ig 向量（可选）：end = body - ig
    if body_pos_frame is not None and ig_frame is not None:
        starts = np.asarray(body_pos_frame)
        ends   = starts - np.asarray(ig_frame)
        scene.add_geometry(_make_line_set(starts, ends), node_name="ig_lines")

    # 7) 导出 or 显示
    if save_path:
        try:
            scene.export(save_path)
            print(f"[viz] saved to {save_path}")
        except Exception as e:
            print("[viz] export failed:", e)

    if show:
        try:
            scene.show(title=title)
        except Exception as e:
            print("[viz] show failed (headless?):", e)
