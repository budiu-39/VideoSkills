import os
import subprocess
from tqdm import tqdm


def batch_hstack_videos(origin_dir, retarget_dir, output_dir):
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取 origin 文件夹下的所有 mp4 文件
    videos = [f for f in os.listdir(origin_dir) if f.endswith('.mp4')]

    print(f"找到 {len(videos)} 个待处理视频...")

    for video_name in tqdm(videos):
        path_left = os.path.join(origin_dir, video_name)
        path_right = os.path.join(retarget_dir, video_name)
        path_out = os.path.join(output_dir, video_name)

        # 检查 retarget 文件夹是否存在同名文件
        if not os.path.exists(path_right):
            print(f"警告: 跳过 {video_name}，在 retarget 文件夹中未找到匹配项。")
            continue

        # 构建 ffmpeg 命令
        # hstack: 横向拼接
        # -c:v libx264: 使用 H.264 编码
        # -crf 23: 质量设置（18-28 之间，越小质量越高）
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖已存在的输出文件
            '-i', path_left,  # 输入 0
            '-i', path_right,  # 输入 1
            '-filter_complex', 'hstack',  # 核心：横向拼接
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',  # 确保兼容性
            path_out
        ]

        # 执行命令 (隐藏控制台输出以保持进度条整洁)
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    # 请修改为你实际的路径
    ORIGIN_PATH = "./renders/origin"
    RETARGET_PATH = "./renders/retarget"
    COMPARE_PATH = "./renders/comparison"

    batch_hstack_videos(ORIGIN_PATH, RETARGET_PATH, COMPARE_PATH)
    print(f"所有视频已合并至: {COMPARE_PATH}")