import torch
from scripts.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState

def pad_skeleton_state(sk_state, padding=10):
    """在 SkeletonState 最前端复制第一帧若干次。padding=0 时不做任何操作。"""
    if padding <= 0:
        return sk_state  # 不需要 padding，直接返回原对象

    root_trans = sk_state.root_translation
    global_rot = sk_state.global_rotation

    # 取第一帧并重复
    first_root = root_trans[0:1].repeat(padding, 1)
    first_global = global_rot[0:1].repeat(padding, 1, 1)

    # 拼接在前端
    new_root = torch.cat([first_root, root_trans], dim=0)
    new_global = torch.cat([first_global, global_rot], dim=0)

    return SkeletonState.from_rotation_and_root_translation(
        sk_state.skeleton_tree,
        new_global,
        new_root,
        is_local=False
    )


if __name__ == "__main__":
    # 简单测试
    # load npy
    import numpy as np
    motion_data = np.load("/home/miku/Documents/VideoSkills/demo/Boxing_kicking_4_clip1.npy", allow_pickle=True).item()

    sk_state = SkeletonState.from_dict(motion_data['skeleton_state'])
    padded_state = pad_skeleton_state(sk_state, padding=3)

    print("Original length:", sk_state.root_translation.shape[0])
    print("Padded length:", padded_state.root_translation.shape[0])


