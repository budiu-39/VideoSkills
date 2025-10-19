import numpy as np
import glob
import joblib

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

    motion_dir = "/home/miku/Documents/VideoSkills/demo/succeed"
    motion_files = sorted(glob.glob(motion_dir + '/*.pkl'))
    trim_length = 10
    for file in motion_files:
        motion_data = joblib.load(file)
        trimmed_motion = trim_motion_sequence(motion_data, trim_length=trim_length)

        out_file = file.replace(".pkl", f"_trim{trim_length}.pkl")
        joblib.dump(trimmed_motion, out_file)
