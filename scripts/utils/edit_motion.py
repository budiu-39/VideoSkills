import numpy as np
import glob
import joblib

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


def trim_motion_sequence(motion_dict, trim_length):
    """裁剪前端指定长度的帧数。trim_length=0 时不做任何操作。"""
    if trim_length <= 0:
        return motion_dict  # 不需要裁剪，直接返回原数据

    trimmed = {}
    for key, value in motion_dict.items():
        if isinstance(value, np.ndarray):
            if value.shape[0] > trim_length:
                trimmed[key] = value[trim_length:]
            else:
                print(f"[WARN] {key} 长度 {value.shape[0]} < trim_length {trim_length}，跳过裁剪")
                trimmed[key] = value
        else:
            # 非 np.ndarray 保持不变
            trimmed[key] = value

    return trimmed


if __name__ == "__main__":
    # 简单测试

    motion_dir = "/demo/succeed"
    motion_files = sorted(glob.glob(motion_dir + '/*.pkl'))
    trim_length = 10
    for file in motion_files:
        motion_data = joblib.load(file)
        trimmed_motion = trim_motion_sequence(motion_data, trim_length=trim_length)

        out_file = file.replace(".pkl", f"_trim{trim_length}.pkl")
        joblib.dump(trimmed_motion, out_file)

if __name__ == "__main__":
    # 简单测试
    # load npy
    import numpy as np
    import glob
    motion_dir = "dataset/smpl_motion/Kungfu"
    motion_out_dir = "dataset/smpl_motion/Kungfu_padded"
    motion_files = sorted(glob.glob(motion_dir + '/*.npy'))
    for motion_file in motion_files:
        motion_data = np.load(motion_file, allow_pickle=True).item()
        # a = SkeletonMotion.from_dict(motion_data)
        sk_state = SkeletonState.from_dict(motion_data)
        padded_state = pad_skeleton_state(sk_state, padding=10)
        motion_data = SkeletonMotion.from_skeleton_state(padded_state, 30)
        out_file = motion_file.replace(motion_dir, motion_out_dir)
        motion_data.to_file(out_file)