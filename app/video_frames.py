"""Extract frames from video files and animated GIFs for vision classification."""

from __future__ import annotations

import io
import os
import shutil
import subprocess
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from PIL import Image

from app.constants import (
    FFMPEG_FRAME_TIMEOUT_SEC,
    GIF_EXTENSION,
    VIDEO_EXTENSIONS,
    VIDEO_FRAME_COUNT,
    VIDEO_FRAGMENT_DECODE_SEC,
    VIDEO_SAMPLE_FRACTIONS,
)

LogFn = Callable[[str], None]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FRAME_CACHE: OrderedDict[tuple[str, int], list[Image.Image]] = OrderedDict()
_FRAME_CACHE_LIMIT = 96


def _noop_log(_msg: str) -> None:
    pass


def _spread_frame_indices(total: int, k: int) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    if k == 1:
        return [0]
    if k >= total:
        return list(range(total))
    return [int(round(i * (total - 1) / (k - 1))) for i in range(k)]


def video_sample_times_sec(duration_sec: float, k: int) -> list[float]:
    """
    Таймкоды для k кадров по VIDEO_SAMPLE_FRACTIONS (разреженно по длине ролика).
    Не требует декодирования файла — только метаданные длительности.
    """
    if k <= 0:
        return []
    eps = 1e-3
    if len(VIDEO_SAMPLE_FRACTIONS) >= k:
        fracs = list(VIDEO_SAMPLE_FRACTIONS[:k])
    elif k == 1:
        fracs = [0.0]
    else:
        fracs = [i / (k - 1) for i in range(k)]
    if duration_sec <= 0.001:
        return [0.0] * k
    out: list[float] = []
    for f in fracs:
        t = float(f) * duration_sec
        t = max(0.0, min(t, max(0.0, duration_sec - eps)))
        out.append(t)
    return out


def resolve_ffmpeg_executable() -> str | None:
    """
    Путь к ffmpeg: PHOTO_AI_SORTER_FFMPEG (файл или папка bin), затем бандл
    в корне проекта (ffmpeg-runtime\\bin), затем PATH и стандартные каталоги Windows.
    """
    raw = os.environ.get("PHOTO_AI_SORTER_FFMPEG", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if p.is_dir():
            p = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if p.is_file():
            return str(p)
    bundled = _PROJECT_ROOT / "ffmpeg-runtime" / "bin" / "ffmpeg.exe"
    if bundled.is_file():
        return str(bundled)
    for exe in _PROJECT_ROOT.glob("ffmpeg-essentials/*/bin/ffmpeg.exe"):
        if exe.is_file():
            return str(exe)
    found = shutil.which("ffmpeg")
    if found:
        return found
    if os.name == "nt":
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pfx86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        for root in (pf, pfx86):
            for rel in (
                Path("ffmpeg") / "bin" / "ffmpeg.exe",
                Path("FFmpeg") / "bin" / "ffmpeg.exe",
            ):
                cand = Path(root) / rel
                if cand.is_file():
                    return str(cand)
        legacy = Path(r"C:\ffmpeg\bin\ffmpeg.exe")
        if legacy.is_file():
            return str(legacy)
    return None


def resolve_ffprobe_executable() -> str | None:
    """Рядом с ffmpeg.exe или PHOTO_AI_SORTER_FFPROBE / PATH."""
    ff = resolve_ffmpeg_executable()
    if ff:
        d = Path(ff).parent
        for name in ("ffprobe.exe", "ffprobe"):
            p = d / name
            if p.is_file():
                return str(p)
    raw = os.environ.get("PHOTO_AI_SORTER_FFPROBE", "").strip().strip('"')
    if raw:
        p = Path(raw)
        if p.is_file():
            return str(p)
    return shutil.which("ffprobe")


def ffprobe_duration_sec(path: Path, on_log: LogFn = _noop_log) -> float | None:
    """Return container duration in seconds, or None if unknown."""
    probe = resolve_ffprobe_executable()
    if not probe:
        on_log("rollback: ffprobe not in PATH")
        return None
    try:
        r = subprocess.run(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            on_log(f"rollback: ffprobe duration failed ({r.returncode})")
            return None
        return float(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired, OSError) as e:
        on_log(f"rollback: ffprobe error: {e!s}")
        return None


def _ffmpeg_frame_at(path: Path, t_sec: float, ffmpeg: str) -> Image.Image | None:
    # -ss перед -i: input seek, без прогона с начала; -t ограничивает объём декода
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_sec:.6f}",
        "-i",
        str(path),
        "-t",
        f"{VIDEO_FRAGMENT_DECODE_SEC:.4f}",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            timeout=FFMPEG_FRAME_TIMEOUT_SEC,
            check=False,
        )
        if r.returncode != 0 or not r.stdout:
            return None
        return Image.open(io.BytesIO(r.stdout)).copy()
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return None


def _ffmpeg_frames(path: Path, n: int, on_log: LogFn) -> list[Image.Image]:
    ffmpeg = resolve_ffmpeg_executable()
    if not ffmpeg:
        on_log("video preview: ffmpeg missing")
        return []
    dur = ffprobe_duration_sec(path, on_log)
    times = video_sample_times_sec(dur if dur is not None else 0.0, n)
    out: list[Image.Image] = []
    if dur is None:
        on_log("video preview: ffprobe unavailable, trying fast frame fallback")
    for t in times:
        im = _ffmpeg_frame_at(path, t, ffmpeg)
        if im is not None:
            out.append(im)
    if not out:
        on_log("video preview: sampled decode failed, trying frame at t=0")
        im = _ffmpeg_frame_at(path, 0.0, ffmpeg)
        if im is not None:
            out.append(im)
    if not out:
        on_log("video preview: ffmpeg decode failed")
    return out


def _opencv_frames(path: Path, n: int, on_log: LogFn) -> list[Image.Image]:
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        on_log("video preview: opencv unavailable")
        return []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        on_log("video preview: opencv could not open file")
        return []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            on_log("video preview: opencv frame count unknown")
            return []
        if n == 1 and total > 0:
            idxs = [total // 2]
        else:
            idxs = _spread_frame_indices(total, n)
        out: list[Image.Image] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out.append(Image.fromarray(rgb))
        return out
    finally:
        cap.release()


def _gif_frames_pillow(path: Path, n: int, on_log: LogFn) -> list[Image.Image]:
    out: list[Image.Image] = []
    try:
        with Image.open(path) as im:
            n_frames = int(getattr(im, "n_frames", 1) or 1)
            if n_frames <= 1:
                im.seek(0)
                return [im.copy()]
            idxs = _spread_frame_indices(n_frames, n)
            for idx in idxs:
                im.seek(idx)
                out.append(im.copy())
            return out
    except Exception as e:
        on_log(f"rollback: GIF Pillow frames: {e!s}")
        return []


def is_animated_gif(path: Path) -> bool:
    if path.suffix.lower() != GIF_EXTENSION:
        return False
    try:
        with Image.open(path) as im:
            return int(getattr(im, "n_frames", 1) or 1) > 1
    except OSError:
        return False


def extract_frames_for_classification(
    path: Path,
    n: int = VIDEO_FRAME_COUNT,
    *,
    on_log: LogFn = _noop_log,
) -> list[Image.Image]:
    """
    Return up to n PIL images (RGB-capable) with fallbacks:
    GIF: Pillow; video: ffmpeg then opencv; shorten list if fewer frames decoded.
    """
    n = max(1, min(n, VIDEO_FRAME_COUNT))
    suf = path.suffix.lower()

    if suf == GIF_EXTENSION:
        return _gif_frames_pillow(path, n, on_log)

    if suf in VIDEO_EXTENSIONS:
        imgs = _ffmpeg_frames(path, n, on_log)
        if len(imgs) >= 1:
            if len(imgs) < n:
                on_log(f"rollback: decoded {len(imgs)} frame(s), wanted {n}")
            return imgs[:n]
        imgs = _opencv_frames(path, n, on_log)
        if len(imgs) >= 1:
            if len(imgs) < n:
                on_log(f"rollback: opencv got {len(imgs)} frame(s) from {n}")
            return imgs[:n]
        on_log("video preview: no frames from ffmpeg or opencv")
        return []

    return []


def preview_frame_for_thumbnail(path: Path, *, on_log: LogFn = _noop_log) -> Image.Image | None:
    """
    One representative frame for UI thumbnails: middle of timeline when duration is known,
    else middle frame index for OpenCV, else first decoded frame.
    """
    suf = path.suffix.lower()
    if suf == GIF_EXTENSION:
        frames = _gif_frames_pillow(path, 1, on_log)
        return frames[0].convert("RGB") if frames else None
    if suf not in VIDEO_EXTENSIONS:
        return None
    ffmpeg = resolve_ffmpeg_executable()
    if ffmpeg:
        dur = ffprobe_duration_sec(path, on_log)
        eps = 0.05
        if dur is not None and dur > eps:
            t = max(0.0, min(dur * 0.5, max(0.0, dur - eps)))
        else:
            t = 0.0
        im = _ffmpeg_frame_at(path, t, ffmpeg)
        if im is None and t > 0:
            im = _ffmpeg_frame_at(path, 0.0, ffmpeg)
        if im is not None:
            return im.convert("RGB")
    imgs = _opencv_frames(path, 1, on_log)
    if imgs:
        return imgs[0].convert("RGB")
    return None


def extract_frames_reduced(
    path: Path,
    n: int,
    *,
    on_log: LogFn = _noop_log,
) -> list[Image.Image]:
    """Try n, then n-1 ... 1 until at least one frame or empty."""
    key = (str(path.resolve()), int(max(1, n)))
    cached = _FRAME_CACHE.get(key)
    if cached is not None:
        _FRAME_CACHE.move_to_end(key)
        return [im.copy() for im in cached]
    for k in range(n, 0, -1):
        frames = extract_frames_for_classification(path, k, on_log=on_log)
        if frames:
            _FRAME_CACHE[key] = [im.copy() for im in frames]
            _FRAME_CACHE.move_to_end(key)
            while len(_FRAME_CACHE) > _FRAME_CACHE_LIMIT:
                _FRAME_CACHE.popitem(last=False)
            return frames
    return []
