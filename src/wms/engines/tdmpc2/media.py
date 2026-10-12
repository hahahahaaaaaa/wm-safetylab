"""Small, explicit video artifacts for real simulation frames."""
from pathlib import Path
import subprocess


def write_video(output, stem, frames, fps, compatibility=True):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    output = Path(output)
    if not frames:
        raise ValueError("Cannot record an empty simulation")
    filename = stem + ".mp4"
    imageio.mimsave(output / filename, frames, fps=fps, macro_block_size=1,
                   codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    poster = stem + "_final.png"
    imageio.imwrite(output / poster, frames[-1])
    result = {"video": filename, "video_poster": poster, "video_frames": len(frames),
              "video_fps": fps, "video_duration_s": len(frames) / fps}
    if compatibility:
        webm = stem + ".webm"
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-n",
                        "-i", str(output / filename), "-c:v", "libvpx", "-b:v", "900k",
                        "-pix_fmt", "yuv420p", "-an", str(output / webm)], check=True, timeout=120)
        result["video_webm"] = webm
    return result
