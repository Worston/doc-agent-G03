"""Stage 1 — classical preprocessing: deskew -> denoise -> binarize -> despeckle.

Speciality E1 (degraded scans) is served here; E2 (multi-column) is served in
``vision/layout.py``. The corpus is a 1918 letterpress book scanned at 300 DPI, so the
degradations are: whole-page skew from the book being laid on the platen, paper speckle
and scanner noise, ink bleed-through from the verso, and page-scale illumination drift
(the gutter side is darker than the fore-edge). A single global threshold cannot serve a
page whose background brightness varies, which is why binarization is local (Sauvola).

``Page`` is a FIXED contract carrying only an image *path*, so a stage that changes pixels
must write new files and hand back Pages pointing at them. Outputs go to
``data/interim/<page_id>.png`` (gitignored, re-derivable from ``data/raw/``).

Augmentation is deliberately NOT done here. This path builds the knowledge base, and
augmenting a KB page would put distorted text into the index; augmentation belongs to the
OCR training loader (``training/datamodule.py``).

Defaults were **measured**, not assumed, on the 40-page human-verified held-out slice
(Tesseract 5, ``--psm 3``, character error rate against ``grading_kit/labels.jsonl``):

    raw (no preprocessing)          0.0914
    deskew + median                 0.0797   <- default
    deskew, no median               0.0885
    + Sauvola binarization          0.0911
    + Otsu binarization             0.0799

Binarization is therefore implemented and selectable but **off by default**: Tesseract
already binarizes internally, so thresholding first only discards greyscale detail it
would have used, and TrOCR expects continuous-tone input. Two further results shaped the
project rather than this file: deskew changed the score by <0.0001 (the Internet Archive
scanner already deskews — measured skew is a median 0.12 deg, max 0.55 deg), and error is
concentrated almost entirely in mixed-layout pages (0.163 vs 0.023 for two-column and
0.027 for single-column), which is a reading-order failure that no pixel filter can fix.
That is the quantified case for the layout stage carrying speciality E2.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import cv2
import numpy as np

from ..contracts import Page
from ..logging_conf import get_logger

log = get_logger(__name__)

_DEFAULTS: dict = {
    "enabled": True,
    "out_dir": "data/interim",
    "deskew": {"max_angle": 3.0, "coarse_step": 0.5, "fine_step": 0.05, "min_angle": 0.05},
    "denoise": {"median_ksize": 3},
    "binarize": {"method": "none", "window": 41, "k": 0.2},
    "despeckle": {"min_area": 12},
    "workers": 0,  # 0 = os.cpu_count() - 1
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, val in (over or {}).items():
        out[key] = _merge(base[key], val) if isinstance(val, dict) and isinstance(base.get(key), dict) else val
    return out


def _profile_score(binary: np.ndarray) -> float:
    """Sharpness of the horizontal ink profile.

    Ink rows sum high, interline gaps sum near zero, so a correctly deskewed page has a
    profile that swings hard between the two. Squared first difference measures that swing
    and peaks at the true angle. Preferred over Hough lines because this book has no ruled
    lines to detect, and over minAreaRect because that fits the page border, not the type.
    """
    profile = binary.sum(axis=1, dtype=np.float64)
    return float((np.diff(profile) ** 2).sum())


def _rotate(img: np.ndarray, angle: float, fill: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(
        img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill
    )


def estimate_skew(gray: np.ndarray, cfg: dict) -> float:
    """Angle in degrees that must be applied to level the text (positive = anticlockwise).

    Searched coarse-to-fine on a downscaled Otsu mask: the profile score is smooth in the
    angle, so a 0.5 deg sweep localises the peak and a 0.05 deg sweep refines it, at ~1/40
    the cost of scanning the full range at final resolution.
    """
    scale = 1000.0 / max(gray.shape)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    _, mask = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    lim, coarse, fine = cfg["max_angle"], cfg["coarse_step"], cfg["fine_step"]
    if not mask.any():
        # A blank or near-blank page (the corpus has 118 sparse pages) scores every angle
        # equally. Without this guard the tie would be broken by array order and the page
        # would be rotated by the full search limit for no reason.
        return 0.0

    def best(candidates: np.ndarray) -> float:
        # Tie-break toward 0: a flat score surface must not produce a large rotation.
        return float(max(candidates, key=lambda a: (_profile_score(_rotate(mask, a, 0)), -abs(a))))

    c = best(np.arange(-lim, lim + coarse / 2, coarse))
    return best(np.clip(np.arange(c - coarse, c + coarse + fine / 2, fine), -lim, lim))


def _sauvola(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    """Sauvola local threshold: t = m * (1 + k * (s / 128 - 1)).

    Chosen over Otsu because the threshold follows the local background, so the darker
    gutter margin is not flooded black; and over plain adaptive-mean because the standard
    deviation term suppresses faint bleed-through (low-contrast regions get a threshold
    below their mean, so ghost text from the verso stays white).
    """
    window += window % 2 == 0  # box filter needs an odd window
    f = np.float32(gray)
    mean = cv2.boxFilter(f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    sq = cv2.boxFilter(f * f, cv2.CV_32F, (window, window), normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    thresh = mean * (1.0 + k * (std / 128.0 - 1.0))
    return np.where(f > thresh, np.uint8(255), np.uint8(0))


def _binarize(gray: np.ndarray, cfg: dict) -> np.ndarray:
    method = cfg["method"]
    if method == "none":
        return gray
    if method == "sauvola":
        return _sauvola(gray, int(cfg["window"]), float(cfg["k"]))
    if method == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        w = int(cfg["window"]) | 1
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w, 10)
    raise ValueError(f"unknown binarize.method {method!r} (sauvola | otsu | adaptive | none)")


def _despeckle(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Drop ink blobs smaller than min_area px.

    A full stop at 10 pt / 300 DPI covers roughly 70 px, so the default 12 px floor clears
    scanner grit and paper flecks without eating punctuation — which matters because the
    held-out transcriptions preserve the original's punctuation verbatim.
    """
    if min_area <= 0:
        return binary
    ink = (binary == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return np.where(keep[labels], np.uint8(0), np.uint8(255))


def preprocess_image(gray: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Full single-page pipeline. Returns the processed image and the skew removed."""
    angle = estimate_skew(gray, cfg["deskew"])
    if abs(angle) >= cfg["deskew"]["min_angle"]:
        # Fill with the page's own paper tone, not white: a white wedge in the corner would
        # read as a hard edge to the layout stage.
        gray = _rotate(gray, angle, int(np.median(gray)))
    else:
        angle = 0.0

    ksize = int(cfg["denoise"]["median_ksize"])
    if ksize > 1:
        # Median before thresholding: it removes isolated specks without softening stroke
        # edges the way a Gaussian would, and cleaner input means fewer local-threshold
        # artefacts to clean up afterwards.
        gray = cv2.medianBlur(gray, ksize | 1)

    out = _binarize(gray, cfg["binarize"])
    if cfg["binarize"]["method"] != "none":
        out = _despeckle(out, int(cfg["despeckle"]["min_area"]))
    return out, angle


def _process_one(args: tuple[str, str, str, dict]) -> tuple[str, float, bool]:
    page_id, src, dst, cfg = args
    gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"unreadable image: {src}")
    out, angle = preprocess_image(gray, cfg)
    if not cv2.imwrite(dst, out):
        raise OSError(f"could not write {dst}")
    return page_id, angle, True


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Deskew, denoise and binarize every page; return Pages pointing at the new images.

    Results are cached in ``out_dir``: a page is reprocessed only when its output is
    missing or older than its source, or when the preprocess config has changed since the
    last run (tracked in ``out_dir/params.json``). Reprocessing all 624 pages costs minutes,
    and every downstream stage re-runs this one.
    """
    pc = _merge(_DEFAULTS, cfg.get("preprocess", {}))
    if not pc["enabled"]:
        log.info("preprocess disabled; passing %d pages through unchanged", len(pages))
        return pages

    out_dir = Path(pc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    params = out_dir / "params.json"
    fingerprint = json.dumps({k: v for k, v in pc.items() if k != "workers"}, sort_keys=True)
    stale_config = not params.is_file() or params.read_text() != fingerprint

    jobs, out_pages = [], []
    for p in pages:
        src, dst = Path(p.image_path), out_dir / f"{p.id}.png"
        if stale_config or not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
            jobs.append((p.id, str(src), str(dst), pc))
        out_pages.append(Page(id=p.id, image_path=str(dst), doc_id=p.doc_id))

    if jobs:
        workers = int(pc["workers"]) or max(1, (os.cpu_count() or 2) - 1)
        log.info("preprocessing %d/%d pages with %d worker(s)", len(jobs), len(pages), workers)
        if workers == 1:
            results = [_process_one(j) for j in jobs]
        else:
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(_process_one, jobs, chunksize=4))
            except BrokenProcessPool:
                # macOS spawns workers, which re-import the parent's __main__. That fails
                # when there is no importable main module — i.e. in a notebook or REPL,
                # which is exactly how the demo runs this. Serial is ~7x slower, not wrong.
                log.warning("process pool unavailable (no importable __main__?); running serially")
                results = [_process_one(j) for j in jobs]
        angles = np.array([a for _, a, _ in results])
        deskewed = int((angles != 0).sum())
        log.info(
            "deskewed %d/%d pages; |angle| median %.2f deg max %.2f deg",
            deskewed,
            len(results),
            float(np.median(np.abs(angles))),
            float(np.abs(angles).max()),
        )
        params.write_text(fingerprint)
    else:
        log.info("preprocess cache hit for all %d pages", len(pages))

    return out_pages
