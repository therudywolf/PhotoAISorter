"""Simplified duplicate detection: exact + perceptual + optional LLM stage."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import imagehash
from PIL import Image

from app.constants import GIF_EXTENSION, MediaScanMode, STILL_IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from app.images import _pil_to_rgb, file_sha256
from app.signature_db import SIG_CACHE_VERSION, SignatureDatabase
from app.video_frames import extract_frames_reduced, is_animated_gif

LogFn = Callable[[str], None]


@dataclass
class DuplicateFinderOptions:
    strictness: Literal["fast", "balanced", "strict", "deep"] = "balanced"
    include_exact: bool = True
    include_perceptual: bool = True
    use_llm_pairs: bool = False
    phash_max_hamming: int = 10
    dhash_max_hamming: int = 12
    phash_video_mean_max: int = 7
    llm_hamming_min: int = 6
    llm_hamming_max: int = 14
    hash_max_side: int = 256
    meta_require_same_dimensions: bool = False
    video_sample_frames: int = 3
    keep_policy: Literal["largest_pixels", "newest_mtime", "largest_file"] = "largest_pixels"
    parallel_workers: int = 3


_PRESETS: dict[str, DuplicateFinderOptions] = {
    "fast": DuplicateFinderOptions(
        strictness="fast",
        include_exact=True,
        include_perceptual=False,
        use_llm_pairs=False,
        phash_max_hamming=8,
        dhash_max_hamming=12,
        phash_video_mean_max=7,
        llm_hamming_min=6,
        llm_hamming_max=14,
        hash_max_side=256,
        meta_require_same_dimensions=False,
        video_sample_frames=3,
    ),
    "balanced": DuplicateFinderOptions(
        strictness="balanced",
        include_exact=True,
        include_perceptual=True,
        use_llm_pairs=False,
        phash_max_hamming=10,
        dhash_max_hamming=12,
        phash_video_mean_max=7,
        llm_hamming_min=6,
        llm_hamming_max=14,
        hash_max_side=256,
        meta_require_same_dimensions=False,
        video_sample_frames=3,
    ),
    "strict": DuplicateFinderOptions(
        strictness="strict",
        include_exact=True,
        include_perceptual=True,
        use_llm_pairs=True,
        phash_max_hamming=8,
        dhash_max_hamming=10,
        phash_video_mean_max=5,
        llm_hamming_min=6,
        llm_hamming_max=12,
        hash_max_side=256,
        meta_require_same_dimensions=False,
        video_sample_frames=5,
    ),
    "deep": DuplicateFinderOptions(
        strictness="deep",
        include_exact=True,
        include_perceptual=True,
        use_llm_pairs=True,
        phash_max_hamming=12,
        dhash_max_hamming=14,
        phash_video_mean_max=6,
        llm_hamming_min=6,
        llm_hamming_max=16,
        hash_max_side=256,
        meta_require_same_dimensions=False,
        video_sample_frames=4,
    ),
}


def options_from_preset(preset: str) -> DuplicateFinderOptions:
    p = _PRESETS.get(preset, _PRESETS["balanced"])
    return DuplicateFinderOptions(**p.__dict__)


def merge_options_from_dict(d: dict[str, Any], base: DuplicateFinderOptions) -> DuplicateFinderOptions:
    o = DuplicateFinderOptions(**base.__dict__)
    o.strictness = str(d.get("strictness", o.strictness)) if str(d.get("strictness", o.strictness)) in _PRESETS else o.strictness
    o.include_exact = bool(d.get("include_exact", o.include_exact))
    o.include_perceptual = bool(d.get("include_perceptual", o.include_perceptual))
    o.use_llm_pairs = bool(d.get("use_llm_pairs", o.use_llm_pairs))
    o.phash_max_hamming = max(0, min(64, int(d.get("phash_max_hamming", o.phash_max_hamming))))
    o.dhash_max_hamming = max(0, min(64, int(d.get("dhash_max_hamming", o.dhash_max_hamming))))
    o.phash_video_mean_max = max(1, min(32, int(d.get("phash_video_mean_max", o.phash_video_mean_max))))
    o.llm_hamming_min = max(0, min(64, int(d.get("llm_hamming_min", o.llm_hamming_min))))
    o.llm_hamming_max = max(0, min(64, int(d.get("llm_hamming_max", o.llm_hamming_max))))
    o.hash_max_side = max(64, min(512, int(d.get("hash_max_side", o.hash_max_side))))
    o.meta_require_same_dimensions = bool(d.get("meta_require_same_dimensions", o.meta_require_same_dimensions))
    o.video_sample_frames = max(1, min(12, int(d.get("video_sample_frames", o.video_sample_frames))))
    o.parallel_workers = max(1, min(4, int(d.get("parallel_workers", o.parallel_workers))))
    kp = str(d.get("keep_policy", o.keep_policy))
    if kp in ("largest_pixels", "newest_mtime", "largest_file"):
        o.keep_policy = kp  # type: ignore[assignment]
    if o.llm_hamming_max < o.llm_hamming_min:
        o.llm_hamming_max = o.llm_hamming_min
    return o


def options_to_dict(o: DuplicateFinderOptions) -> dict[str, Any]:
    return {
        "strictness": o.strictness,
        "include_exact": o.include_exact,
        "include_perceptual": o.include_perceptual,
        "use_llm_pairs": o.use_llm_pairs,
        "phash_max_hamming": o.phash_max_hamming,
        "dhash_max_hamming": o.dhash_max_hamming,
        "phash_video_mean_max": o.phash_video_mean_max,
        "llm_hamming_min": o.llm_hamming_min,
        "llm_hamming_max": o.llm_hamming_max,
        "hash_max_side": o.hash_max_side,
        "meta_require_same_dimensions": o.meta_require_same_dimensions,
        "video_sample_frames": o.video_sample_frames,
        "keep_policy": o.keep_policy,
        "parallel_workers": max(1, min(4, o.parallel_workers)),
    }


@dataclass
class FileDupInfo:
    path: Path
    path_norm: str
    size_bytes: int
    mtime_ns: int
    width: int | None
    height: int | None
    sha256: str | None
    phash: imagehash.ImageHash | None
    dhash: imagehash.ImageHash | None
    phash_frames: list[imagehash.ImageHash] | None = None


@dataclass
class DuplicateGroup:
    paths: list[Path]
    suggested_keep: Path
    is_exact: bool = False


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def _norm(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _stat(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
    except OSError:
        return None


def iter_dup_scan_paths(root: Path, mode: MediaScanMode) -> list[Path]:
    from app.worker import iter_media_files

    return iter_media_files(root, mode)


def _load_image(path: Path, options: DuplicateFinderOptions, on_log: LogFn) -> Image.Image | None:
    suf = path.suffix.lower()
    try:
        if suf in STILL_IMAGE_EXTENSIONS:
            with Image.open(path) as im:
                return im.copy()
        if suf == GIF_EXTENSION:
            if is_animated_gif(path):
                fs = extract_frames_reduced(path, max(1, options.video_sample_frames), on_log=on_log)
                return fs[0] if fs else None
            with Image.open(path) as im:
                return im.copy()
        if suf in VIDEO_EXTENSIONS:
            fs = extract_frames_reduced(path, max(1, options.video_sample_frames), on_log=on_log)
            return fs[0] if fs else None
    except OSError as e:
        on_log(f"open fail {path}: {e!s}")
    return None


def _load_frames_for_signature(path: Path, options: DuplicateFinderOptions, on_log: LogFn) -> tuple[list[Image.Image], bool]:
    """Возвращает кадры для подписи; второй флаг — нужен ли набор pHash по нескольким кадрам (видео / GIF)."""
    suf = path.suffix.lower()
    n = max(3, options.video_sample_frames)
    try:
        if suf in STILL_IMAGE_EXTENSIONS:
            with Image.open(path) as im:
                return [im.copy()], False
        if suf == GIF_EXTENSION:
            if is_animated_gif(path):
                fs = extract_frames_reduced(path, n, on_log=on_log)
                return (fs, True) if fs else ([], False)
            with Image.open(path) as im:
                return [im.copy()], False
        if suf in VIDEO_EXTENSIONS:
            fs = extract_frames_reduced(path, n, on_log=on_log)
            return (fs, True) if fs else ([], False)
    except OSError as e:
        on_log(f"open fail {path}: {e!s}")
    return [], False


def _parse_phash_frames_json(raw: str | None) -> list[imagehash.ImageHash] | None:
    if not raw:
        return None
    try:
        arr = json.loads(str(raw))
        if not isinstance(arr, list) or len(arr) < 2:
            return None
        return [imagehash.hex_to_hash(str(x)) for x in arr]
    except Exception:
        return None


def _hashes(im: Image.Image, max_side: int) -> tuple[imagehash.ImageHash, imagehash.ImageHash]:
    im = _pil_to_rgb(im)
    w, h = im.size
    long = max(w, h)
    if long > max_side:
        k = max_side / float(long)
        im = im.resize((max(1, int(round(w * k))), max(1, int(round(h * k)))), Image.Resampling.LANCZOS)
    return imagehash.phash(im), imagehash.dhash(im)


def _hamming(a: imagehash.ImageHash | None, b: imagehash.ImageHash | None) -> int:
    if a is None or b is None:
        return 9999
    return int(a - b)


def _path_needs_multiframe_phash(path: Path) -> bool:
    suf = path.suffix.lower()
    if suf in VIDEO_EXTENSIONS:
        return True
    if suf == GIF_EXTENSION:
        return is_animated_gif(path)
    return False


def _multi_frame_phash_mean_dist(
    fa: list[imagehash.ImageHash], fb: list[imagehash.ImageHash]
) -> float:
    if not fa or not fb:
        return 999.0
    s1 = sum(min(int(x - y) for y in fb) for x in fa)
    s2 = sum(min(int(x - y) for x in fa) for y in fb)
    return (s1 / len(fa) + s2 / len(fb)) / 2.0


def _perceptual_match_records(a: FileDupInfo, b: FileDupInfo, options: DuplicateFinderOptions) -> bool:
    """Согласование pHash (+ dHash для фото) или среднее по кадрам для видео."""
    if not _likely_candidate(a, b):
        return False
    fa_m = a.phash_frames if a.phash_frames and len(a.phash_frames) >= 2 else None
    fb_m = b.phash_frames if b.phash_frames and len(b.phash_frames) >= 2 else None
    if fa_m is not None or fb_m is not None:
        eff_a = fa_m if fa_m is not None else ([a.phash] if a.phash else [])
        eff_b = fb_m if fb_m is not None else ([b.phash] if b.phash else [])
        if not eff_a or not eff_b:
            return False
        return _multi_frame_phash_mean_dist(eff_a, eff_b) <= float(options.phash_video_mean_max)
    hp = _hamming(a.phash, b.phash)
    if hp > options.phash_max_hamming:
        return False
    if a.dhash is not None and b.dhash is not None:
        if _hamming(a.dhash, b.dhash) > options.dhash_max_hamming:
            return False
    return True


def _dedupe_records_by_path(records: list[FileDupInfo]) -> list[FileDupInfo]:
    seen: set[str] = set()
    out: list[FileDupInfo] = []
    for r in records:
        if r.path_norm in seen:
            continue
        seen.add(r.path_norm)
        out.append(r)
    return out


def _likely_candidate(a: FileDupInfo, b: FileDupInfo) -> bool:
    aw, ah = a.width or 0, a.height or 0
    bw, bh = b.width or 0, b.height or 0
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return True
    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    area_ratio = max(area_a, area_b) / float(min(area_a, area_b))
    if area_ratio > 3.5:
        return False
    ar_a = aw / float(ah)
    ar_b = bw / float(bh)
    if max(ar_a, ar_b) / max(0.0001, min(ar_a, ar_b)) > 2.0:
        return False
    return True


def collect_llm_verification_pairs(
    records: list[FileDupInfo],
    options: DuplicateFinderOptions,
    max_pairs: int,
) -> list[tuple[FileDupInfo, FileDupInfo]]:
    """
    Candidate pairs for LLM duplicate verification.

    Uses the same coarse size buckets as perceptual grouping (not a full O(n²) scan over
    all files), so large libraries stay tractable. Pairs that only match across buckets
    are skipped (rare vs. same-bucket near-duplicates).
    """
    pairs: list[tuple[FileDupInfo, FileDupInfo]] = []
    if max_pairs <= 0 or len(records) < 2:
        return pairs
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.phash is None:
            continue
        if options.meta_require_same_dimensions:
            key = (r.width or -1, r.height or -1)
        else:
            key = ((r.width or -1) // 32, (r.height or -1) // 32)
        buckets[key].append(i)
    for key in sorted(buckets.keys()):
        idxs = buckets[key]
        m = len(idxs)
        for a in range(m):
            for b in range(a + 1, m):
                ia, ib = idxs[a], idxs[b]
                ai, bi = records[ia], records[ib]
                if not _likely_candidate(ai, bi):
                    continue
                if ai.sha256 and bi.sha256 and ai.sha256 == bi.sha256:
                    continue
                h = int(ai.phash - bi.phash)
                if options.llm_hamming_min <= h <= options.llm_hamming_max:
                    pairs.append((ai, bi))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def compute_one_signature(
    path: Path,
    sig_db: SignatureDatabase,
    options: DuplicateFinderOptions,
    *,
    force_recompute: bool,
    on_log: LogFn,
) -> FileDupInfo | None:
    path_norm = _norm(path)
    st = _stat(path)
    if st is None:
        return None
    size_b, mtime_ns = st
    row = None if force_recompute else sig_db.get_row(path_norm)
    needs_mf = _path_needs_multiframe_phash(path)
    pj: str | None = None
    frames_json_ok = False
    if row is not None:
        try:
            pj = row["phash_frames_json"]
        except (KeyError, IndexError, TypeError):
            pj = None
        frames_json_ok = bool(pj and str(pj).strip())

    if (
        row is not None
        and int(row["mtime_ns"]) == mtime_ns
        and int(row["size_bytes"]) == size_b
        and str(row["sig_version"] or "") == SIG_CACHE_VERSION
        and row["phash_hex"]
        and (not needs_mf or frames_json_ok)
    ):
        try:
            ph = imagehash.hex_to_hash(str(row["phash_hex"]))
            dh = imagehash.hex_to_hash(str(row["dhash_hex"])) if row["dhash_hex"] else None
        except Exception:
            ph = None
            dh = None
        if ph is not None:
            pf = _parse_phash_frames_json(str(pj) if frames_json_ok else None)
            return FileDupInfo(
                path=path,
                path_norm=path_norm,
                size_bytes=size_b,
                mtime_ns=mtime_ns,
                width=int(row["width"]) if row["width"] is not None else None,
                height=int(row["height"]) if row["height"] is not None else None,
                sha256=str(row["sha256"]) if row["sha256"] else None,
                phash=ph,
                dhash=dh,
                phash_frames=pf,
            )

    frames, multi = _load_frames_for_signature(path, options, on_log)
    if not frames:
        return None
    im0 = frames[0]
    width, height = im0.size
    sha = None
    if options.include_exact:
        try:
            sha = file_sha256(path)
        except OSError:
            sha = None
    ph_list: list[imagehash.ImageHash] = []
    dh0: imagehash.ImageHash | None = None
    for i, im in enumerate(frames):
        ph_i, dh_i = _hashes(im, options.hash_max_side)
        ph_list.append(ph_i)
        if i == 0:
            dh0 = dh_i
    ph = ph_list[0]
    phash_frames_json: str | None = None
    phash_frames: list[imagehash.ImageHash] | None = None
    if multi and len(ph_list) >= 2:
        phash_frames_json = json.dumps([str(x) for x in ph_list])
        phash_frames = ph_list
    sig_db.upsert_signature(path_norm, mtime_ns, size_b, width, height, sha, str(ph), str(dh0) if dh0 else None, phash_frames_json)
    return FileDupInfo(
        path, path_norm, size_b, mtime_ns, width, height, sha, ph, dh0, phash_frames=phash_frames
    )


def _suggest(paths: list[Path], policy: str, infos: dict[str, FileDupInfo]) -> Path:
    def pixels(p: Path) -> int:
        r = infos.get(_norm(p))
        return (r.width or 0) * (r.height or 0) if r else 0

    def mtime(p: Path) -> int:
        r = infos.get(_norm(p))
        return r.mtime_ns if r else 0

    def size(p: Path) -> int:
        r = infos.get(_norm(p))
        return r.size_bytes if r else 0

    if policy == "newest_mtime":
        return max(paths, key=lambda p: (mtime(p), pixels(p), size(p)))
    if policy == "largest_file":
        return max(paths, key=lambda p: (size(p), pixels(p), mtime(p)))
    return max(paths, key=lambda p: (pixels(p), size(p), mtime(p)))


def build_groups_from_records(
    records: list[FileDupInfo],
    options: DuplicateFinderOptions,
    *,
    on_log: LogFn | None = None,
    llm_pair_fn: Callable[[Path, Path], bool] | None = None,
    llm_pair_decisions: dict[tuple[str, str], bool] | None = None,
) -> list[DuplicateGroup]:
    _ = on_log
    records = _dedupe_records_by_path(records)
    n = len(records)
    if n < 2:
        return []
    uf = UnionFind(n)

    if options.include_exact:
        by_sha: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            if r.sha256:
                by_sha[r.sha256].append(i)
        for idxs in by_sha.values():
            if len(idxs) > 1:
                h = idxs[0]
                for i in idxs[1:]:
                    uf.union(h, i)

    if options.include_perceptual:
        buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            if r.phash is None:
                continue
            if options.meta_require_same_dimensions:
                key = (r.width or -1, r.height or -1)
            else:
                key = ((r.width or -1) // 32, (r.height or -1) // 32)
            buckets[key].append(i)
        max_bucket = max((len(v) for v in buckets.values()), default=0)
        if on_log and max_bucket > 6000:
            on_log(
                f"Группировка pHash: самая большая корзина размеров — {max_bucket} файлов; "
                "сравнение пар внутри неё может занять много времени."
            )
        for idxs in buckets.values():
            m = len(idxs)
            for a in range(m):
                for b in range(a + 1, m):
                    ia, ib = idxs[a], idxs[b]
                    if _perceptual_match_records(records[ia], records[ib], options):
                        uf.union(ia, ib)

    if options.use_llm_pairs:
        if llm_pair_decisions is not None:
            # O(K) over verified pairs only — avoids O(n²) scans + per-pair DB lookups after LLM stage.
            idx_by_resolved = {str(records[i].path.resolve()): i for i in range(n)}
            for (pa, pb), is_dup in llm_pair_decisions.items():
                if not is_dup:
                    continue
                ia = idx_by_resolved.get(pa)
                ib = idx_by_resolved.get(pb)
                if ia is None or ib is None:
                    continue
                if uf.find(ia) == uf.find(ib):
                    continue
                uf.union(ia, ib)
        elif llm_pair_fn is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if uf.find(i) == uf.find(j):
                        continue
                    if not _likely_candidate(records[i], records[j]):
                        continue
                    h = _hamming(records[i].phash, records[j].phash)
                    if options.llm_hamming_min <= h <= options.llm_hamming_max:
                        if llm_pair_fn(records[i].path, records[j].path):
                            uf.union(i, j)

    roots: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        roots[uf.find(i)].append(i)

    infos = {r.path_norm: r for r in records}
    out: list[DuplicateGroup] = []
    for idxs in roots.values():
        if len(idxs) < 2:
            continue
        uniq_paths: dict[str, Path] = {}
        for i in idxs:
            uniq_paths.setdefault(records[i].path_norm, records[i].path)
        paths = sorted(uniq_paths.values(), key=lambda p: str(p).lower())
        if len(paths) < 2:
            continue
        shas = {records[i].sha256 for i in idxs if records[i].sha256}
        out.append(
            DuplicateGroup(
                paths=paths,
                suggested_keep=_suggest(paths, options.keep_policy, infos),
                is_exact=bool(options.include_exact) and len(shas) == 1 and bool(shas),
            )
        )
    out.sort(key=lambda g: (-len(g.paths), str(g.paths[0]).lower()))
    return out


def regroup_from_cached_records(
    records: list[FileDupInfo],
    options: DuplicateFinderOptions,
    *,
    on_log: LogFn | None = None,
    llm_pair_fn: Callable[[Path, Path], bool] | None = None,
    llm_pair_decisions: dict[tuple[str, str], bool] | None = None,
) -> list[DuplicateGroup]:
    return build_groups_from_records(
        records,
        options,
        on_log=on_log,
        llm_pair_fn=llm_pair_fn,
        llm_pair_decisions=llm_pair_decisions,
    )


def load_records_from_db(paths: list[Path], sig_db: SignatureDatabase, options: DuplicateFinderOptions) -> list[FileDupInfo]:
    out: list[FileDupInfo] = []
    for p in paths:
        row = sig_db.get_row(_norm(p))
        st = _stat(p)
        if row is None or st is None or not row["phash_hex"]:
            continue
        size_b, mtime_ns = st
        if int(row["mtime_ns"]) != mtime_ns or int(row["size_bytes"]) != size_b:
            continue
        try:
            ph = imagehash.hex_to_hash(str(row["phash_hex"]))
            dh = imagehash.hex_to_hash(str(row["dhash_hex"])) if row["dhash_hex"] else None
        except Exception:
            continue
        try:
            pj = row["phash_frames_json"]
        except (KeyError, IndexError, TypeError):
            pj = None
        pf = _parse_phash_frames_json(str(pj) if pj else None)
        out.append(
            FileDupInfo(
                path=p,
                path_norm=_norm(p),
                size_bytes=size_b,
                mtime_ns=mtime_ns,
                width=int(row["width"]) if row["width"] is not None else None,
                height=int(row["height"]) if row["height"] is not None else None,
                sha256=str(row["sha256"]) if row["sha256"] else None,
                phash=ph,
                dhash=dh,
                phash_frames=pf,
            )
        )
    return out


def hamming_matrix_stats(records: list[FileDupInfo]) -> tuple[int, int]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.phash is None:
            continue
        buckets[((r.width or -1) // 32, (r.height or -1) // 32)].append(i)
    max_sz = max((len(v) for v in buckets.values()), default=0)
    pairs = sum((len(v) * (len(v) - 1)) // 2 for v in buckets.values() if len(v) > 1)
    return pairs, max_sz
