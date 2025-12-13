import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio


def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4, kinetic_chain='smpl'):
    matplotlib.use('Agg')

    joints, out_name, title = args

    # (T, J, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    # ---------------------------------------------------------
    # 1. 坐标轴修正 (Axis Correction)
    # ---------------------------------------------------------
    # Sim (Mujoco) 数据通常是 Z-up (sim 或 sim_22)。
    # Matplotlib 的 3D 绘图假设 Y-up (XZ 是地面)。
    # 因此我们需要交换 Y 和 Z 轴: (x, y, z) -> (x, z, y)
    if kinetic_chain in ['sim', 'sim_22']:
        data = data[..., [0, 2, 1]]

    # ---------------------------------------------------------
    # 2. 定义骨骼链 (Define Kinetic Chain)
    # ---------------------------------------------------------
    nb_joints = data.shape[1]

    if kinetic_chain == 'smpl':
        # 原始 HumanML3D / KIT 的 SMPL 定义
        smpl_kinetic_chain = [[0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4], [3, 5, 6, 7],
                              [3, 8, 9, 10]] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                                                                      [0, 3, 6, 9, 12, 15],
                                                                      [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
    elif kinetic_chain == 'sim':
        # SMPL Mujoco 定义 (24关节，包含手)
        # Indices:
        # 0:Pelvis, 1:L_Hip, ... 9:Torso, 10:Spine, 11:Chest ... 14:L_Thorax ...
        smpl_kinetic_chain = [
            [0, 9, 10, 11, 12, 13],  # Spine: Pelvis -> Torso -> Spine -> Chest -> Neck -> Head
            [0, 5, 6, 7, 8],  # R Leg: Pelvis -> R_Hip -> ... -> R_Toe
            [0, 1, 2, 3, 4],  # L Leg: Pelvis -> L_Hip -> ... -> L_Toe
            [11, 19, 20, 21, 22, 23],  # R Arm: Chest -> R_Thorax -> ... -> R_Hand
            [11, 14, 15, 16, 17, 18]  # L Arm: Chest -> L_Thorax -> ... -> L_Hand
        ]
    elif kinetic_chain == 'sim_22':
        # SMPL Mujoco 定义 (22关节，已移除 L_Hand(18) 和 R_Hand(23))
        # 索引位移：L_Hand 后面的关节索引全部 -1
        smpl_kinetic_chain = [
            [0, 9, 10, 11, 12, 13],  # Spine: Pelvis -> Torso -> Spine -> Chest -> Neck -> Head (不变)
            [0, 5, 6, 7, 8],  # R Leg: Pelvis -> R_Hip -> ... -> R_Toe (不变)
            [0, 1, 2, 3, 4],  # L Leg: Pelvis -> L_Hip -> ... -> L_Toe (不变)

            # R Arm (索引前移): Chest(11) -> R_Thorax(18) -> R_Shoulder(19) -> R_Elbow(20) -> R_Wrist(21)
            [11, 18, 19, 20, 21],

            # L Arm (末端截断): Chest(11) -> L_Thorax(14) -> L_Shoulder(15) -> L_Elbow(16) -> L_Wrist(17)
            [11, 14, 15, 16, 17]
        ]
    else:
        raise ValueError(f"Unknown kinetic_chain: {kinetic_chain}")

    limits = 1000 if nb_joints == 21 else 2
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']
    frame_number = data.shape[0]

    # 对齐地面 (Height Alignment)
    # 这里的 data 已经是 Y-up 了，所以 index 1 是高度
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]  # 根节点在地面的轨迹 (X, Z)

    # 以第一帧根节点为中心 (Centering)
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim(-limits, limits)
            ax.set_ylim(-limits, limits)
            ax.set_zlim(0, limits)
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [
                [minx, miny, minz],
                [minx, miny, maxz],
                [maxx, miny, maxz],
                [maxx, miny, minz]
            ]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.), dpi=96) if nb_joints == 21 else plt.figure(figsize=(10, 10),
                                                                                                    dpi=96)
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        _clear_axes(ax)
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5

        # 绘制地面和轨迹
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0],
                     0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
                      color='blue')

        # 绘制骨骼
        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0

            # 安全检查：只绘制存在的关节
            valid_chain = [k for k in chain if k < nb_joints]

            if len(valid_chain) > 1:
                ax.plot3D(data[index, valid_chain, 0],
                          data[index, valid_chain, 1],
                          data[index, valid_chain, 2],
                          linewidth=linewidth, color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()
        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def _clear_axes(ax):
    for ln in list(ax.lines):
        ln.remove()
    for coll in list(ax.collections):
        coll.remove()
    for txt in list(ax.texts):
        txt.remove()
    for p in list(ax.patches):
        p.remove()


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None, fps=30, kinetic_chain='smpl'):
    """
    Args:
        smpl_joints_batch: (B, T, J, 3) 关节数据
        title_batch: 标题列表
        outname: 输出文件名列表
        fps: 帧率
        kinetic_chain:
            'smpl' (默认),
            'sim' (Mujoco Z-up 24 joints),
            'sim_22' (Mujoco Z-up 22 joints, no hands)
    """
    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        # 将 kinetic_chain 参数传递给 plot_3d_motion
        out.append(plot_3d_motion(
            [smpl_joints_batch[i], None, title_batch[i] if title_batch is not None else None],
            kinetic_chain=kinetic_chain
        ))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), fps=fps)
    out = torch.stack(out, axis=0)
    return out