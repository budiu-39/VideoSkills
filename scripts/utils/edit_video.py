import os
import shutil

# === 修改这里 ===
mp4_dir = "/mnt/lustre/work/ponsmoll/pba936/VideoSkills/kungfu_272_video"   # 存放所有 mp4 的文件夹路径
success_txt = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/kungfu/new_failed_keys.txt"  # 成功样本列表
fail_txt = "/mnt/lustre/work/ponsmoll/pba936/MotionStreamer/data/kungfu/new_success_keys.txt"     # 失败样本列表
# =================

success_dir = os.path.join(mp4_dir, "success_videos")
fail_dir = os.path.join(mp4_dir, "failed_videos")
os.makedirs(success_dir, exist_ok=True)
os.makedirs(fail_dir, exist_ok=True)

def read_names(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

success_names = set(read_names(success_txt))
fail_names = set(read_names(fail_txt))

for file in os.listdir(mp4_dir):
    if not file.endswith(".mp4"):
        continue
    base = os.path.splitext(file)[0]
    src = os.path.join(mp4_dir, file)
    if base in success_names:
        shutil.move(src, os.path.join(success_dir, file))
    elif base in fail_names:
        shutil.move(src, os.path.join(fail_dir, file))

print("✅ 分类完成！成功视频放在:", success_dir)
print("❌ 失败视频放在:", fail_dir)

import sys
from moviepy.editor import VideoFileClip

def trim_by_seconds(inp, start_s, dur_s, outp):
    start_s = float(start_s)
    dur_s = float(dur_s)
    end_s = start_s + dur_s
    with VideoFileClip(inp) as clip:
        sub = clip.subclip(start_s, min(end_s, clip.duration))
        # 默认重编码：更稳，兼容性好
        # 你也可以设置参数，比如码率、帧率、音频：
        sub.write_videofile(outp, codec="libx264", audio_codec="aac")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python trim_moviepy.py <input> <start_seconds> <duration_seconds> <output>")
        sys.exit(1)
    _, inp, start, dur, outp = sys.argv
    trim_by_seconds(inp, start, dur, outp)