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