"""Encode generated PNG frames without changing inference."""
import subprocess
from PIL import Image


def encode_animation(pattern, count, fps, destination, output_format="mp4", quality=80):
    """Quality is codec-relative compression quality, not a model quality score."""
    if output_format not in ("mp4", "webm", "webp"):
        raise ValueError("Unsupported output format")
    if not 1 <= quality <= 100 or count < 1 or fps <= 0:
        raise ValueError("Invalid output quality, frame count or FPS")
    if output_format == "webp":
        frames = []
        try:
            for i in range(1, count + 1):
                with Image.open(pattern % i) as image:
                    frames.append(image.convert("RGB"))
            # WebP stores integer milliseconds. Round cumulative timestamps to
            # avoid accumulating error at rates such as 24 FPS.
            durations = [round((i + 1) * 1000 / fps) - round(i * 1000 / fps)
                         for i in range(count)]
            frames[0].save(destination, format="WEBP", save_all=True,
                           append_images=frames[1:], duration=durations,
                           loop=0, quality=quality, lossless=False, method=4)
        finally:
            for frame in frames:
                frame.close()
        return
    # 80 preserves the existing MP4 CRF 15 default. 100 is the
    # highest setting, not a promise of lossless RGB (YUV420 subsampling).
    crf = round((100 - quality) * (0.75 if output_format == "mp4" else 0.63))
    crf = min(51 if output_format == "mp4" else 63, crf)
    codec = (["-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
              "-movflags", "+faststart"] if output_format == "mp4" else
             ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
              "-deadline", "good", "-cpu-used", "4"])
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                    "-framerate", str(fps), "-start_number", "1", "-i", pattern,
                    "-frames:v", str(count), *codec, "-pix_fmt", "yuv420p",
                    str(destination)], check=True, stdin=subprocess.DEVNULL)
