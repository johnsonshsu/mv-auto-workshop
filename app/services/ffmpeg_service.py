"""FFmpeg 服務：影格數查詢、精確裁切、取最後一幀、接合、配音軌、檢查。"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from app.core.models import UserFacingError

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _run(args: list[str], what: str, cwd: Path | None = None, timeout: float = 3600.0) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                cwd=str(cwd) if cwd else None, timeout=timeout, creationflags=_NO_WINDOW)
    except FileNotFoundError:
        raise UserFacingError(f"找不到 {args[0]}", "請安裝 FFmpeg 並加入 PATH")
    except subprocess.TimeoutExpired:
        raise UserFacingError(f"{what}逾時", "請再試一次")
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-4:])
        raise UserFacingError(f"{what}失敗：{tail}", "詳細內容已寫在記錄區")
    return result


def probe_video(path: Path) -> dict:
    """回傳 {frames, fps, width, height, codec, pix_fmt}；frames 是實際解碼數出來的影格數。"""
    result = _run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                   "stream=nb_read_frames,r_frame_rate,width,height,codec_name,pix_fmt", "-of", "default=nw=1",
                   str(path)], f"讀取影片資訊（{path.name}）")
    info = dict(line.split("=", 1) for line in result.stdout.strip().splitlines() if "=" in line)
    numerator, _, denominator = info.get("r_frame_rate", "0/1").partition("/")
    return {"frames": int(info.get("nb_read_frames", 0)), "fps": float(numerator) / float(denominator or 1),
            "width": int(info.get("width", 0)), "height": int(info.get("height", 0)),
            "codec": info.get("codec_name", ""), "pix_fmt": info.get("pix_fmt", "")}


def trim_segment(source: Path, dest: Path, drop_head: int, keep: int, fps: float, crf: int = 16,
                 width: int = 0, height: int = 0) -> int:
    """丟掉開頭 drop_head 幀後保留 keep 幀，重新編碼成統一規格（H.264、yuv420p、固定 fps、無音軌）。

    來源影格不夠時，以最後一幀補足。回傳補了幾幀。
    有給 width、height 而來源尺寸不同時（模型輸出的尺寸可能被取整），縮放成指定尺寸，各段才接得起來。
    """
    info = probe_video(source)
    available = info["frames"] - drop_head
    pad = max(keep - available, 0)
    filters = [f"select='between(n\\,{drop_head}\\,{drop_head + keep - 1})'", "setpts=N/FRAME_RATE/TB"]
    if pad:
        filters.append(f"tpad=stop_mode=clone:stop={pad}")
    if width and height and (info["width"], info["height"]) != (width, height):
        filters.append(f"scale={width}:{height}:flags=lanczos,setsar=1")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", ",".join(filters), "-r", f"{fps:g}",
          "-frames:v", str(keep), "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
          "-pix_fmt", "yuv420p", str(dest)], f"裁切影片（{source.name}）")
    return pad


def tail_clip(source: Path, dest: Path, frames: int, fps: float, crf: int = 14) -> None:
    """把影片最後 frames 幀另存成一支短片（不足就整支都取），給模型當接續動作的參考。"""
    start = max(probe_video(source)["frames"] - frames, 0)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", f"trim=start_frame={start},setpts=PTS-STARTPTS",
          "-r", f"{fps:g}", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
          str(dest)], "擷取上一段的尾端")


def extract_frame(source: Path, dest: Path, index: int) -> None:
    """把第 index 幀（從 0 起）存成無損 PNG。"""
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", f"select='eq(n\\,{index})'",
          "-frames:v", "1", "-update", "1", str(dest)], f"取出影格（{source.name}）")
    if not dest.exists():
        raise UserFacingError(f"無法從 {source.name} 取出第 {index + 1} 幀", "這一段影片可能不完整，請重跑這一段")


def concat_with_audio(segments: list[Path], audio: Path, dest: Path, work_dir: Path) -> None:
    """依序接合各段（不重新編碼），捨棄各段自帶的音軌，配上完整歌曲（AAC 48 kHz）。"""
    listing = work_dir / "concat_list.txt"
    # concat 清單用相對於 work_dir 的檔名，避開路徑中的空白、中文與跳脫問題
    listing.write_text("".join(f"file '{p.name}'\n" for p in segments), encoding="utf-8")
    for segment in segments:
        if segment.parent != work_dir:
            raise UserFacingError("要接合的影片必須都在同一個資料夾", "這是程式錯誤，請回報")
    silent = work_dir / "concat_silent.mp4"
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", listing.name, "-c", "copy", "-an",
          silent.name], "接合影片", cwd=work_dir)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(silent), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
          "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-movflags", "+faststart", str(dest)],
         "配回音軌")
    silent.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)


def detect_black(path: Path, min_seconds: float = 0.03) -> list[tuple[float, float]]:
    """回傳影片中黑畫面的區間（秒）。"""
    result = _run(["ffmpeg", "-v", "info", "-i", str(path), "-vf", f"blackdetect=d={min_seconds}:pix_th=0.05",
                   "-an", "-f", "null", "-"], "檢查黑畫面")
    return [(float(a), float(b)) for a, b in
            re.findall(r"black_start:([\d.]+)\s+black_end:([\d.]+)", result.stderr)]


def duration_of(path: Path) -> float:
    result = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                  f"讀取長度（{path.name}）")
    return float(result.stdout.strip() or 0)
