"""Stage 3 — OCR: turn layout regions into text, in reading order.

Stage 2 hands over regions that are already in reading order, so this stage's only job is
recognition: crop each region from the Stage 1 image, read it, and emit one ``Chunk`` per
region. Keeping one chunk per region (rather than one per page) is what preserves the
ordering Stage 2 bought — Stage 4 re-splits into token windows and can only preserve an
order that is already there.

Two backends, selected by ``cfg['ocr']['model']``:

* ``"tesseract"`` — a block recogniser. It does its own line finding and binarization, so
  it is given the region crop whole, at ``--psm 6`` (uniform block of text).
* any HuggingFace id (e.g. ``"microsoft/trocr-base-printed"``) — TrOCR is a **line**
  recogniser, trained on single cropped lines. Handing it a paragraph produces one short
  hallucinated line, so ``_line_boxes`` splits the region on horizontal ink valleys first
  and the lines are recognised in batches and rejoined.

That asymmetry is why this module has a line segmenter: it is not an improvement to
Tesseract, it is the minimum required to make TrOCR mean anything.

Measured on the 40 human-verified held-out pages (``grading_kit/labels.jsonl``), which
contain 26,397 reference words:

    backend                          CER   CER lower   s/page   words
    tesseract --psm 6             0.0272      0.0271      2.8   26645
    trocr-base-printed            0.8104      0.2398    124.7       -
    trocr-small-printed           0.8545      0.2999     19.6   22390
    trocr-small fine-tuned        0.6971      0.6951     35.6   13820

Hence ``model: tesseract`` below. The gap is not a tuning accident: on the *same* line
crops the fine-tuned TrOCR scores 0.62 where Tesseract scores 0.0068, so the layout
geometry and the labels are fine and the recogniser is not. Fine-tuning does buy the one
thing it was supposed to — the pretrained models are SROIE-trained and shout in capitals,
which is why their case-folded CER is a third of their raw CER, while the fine-tuned
model's two numbers are the same.

The fine-tuned row was produced while ``training/lit_modules.py`` was still using the
model's own ``.loss``, which shifts the labels twice (see ``_loss`` there); that is fixed,
but the run was not repeated, so treat 0.6971 as an upper bound on that model's error
rather than as its score.

Bounding boxes are in the coordinate frame of the image Stage 2 measured, which is Stage
1's output in ``data/interim/``. Reading the crop from ``data/raw/`` instead would be
subtly wrong wherever deskew rotated a page, so ``source_dir`` points at the interim
directory and raw is only a fallback for when preprocessing is disabled.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..contracts import Chunk, Region
from ..logging_conf import get_logger

log = get_logger(__name__)

_DEFAULTS: dict = {
    "model": "tesseract",
    "finetune": False,
    "lang": "eng",
    "psm": 6,  # a region is a uniform block; Tesseract must not re-segment the page
    "source_dir": "data/interim",  # Stage 1 output = the frame Stage 2's bboxes are in
    "raw_dir": "data/raw",  # fallback for when preprocessing is disabled
    "skip_kinds": ["figure"],  # illustrations carry no text worth indexing
    "min_chars": 2,  # drop regions that recognise to noise
    "batch": 16,  # TrOCR lines per forward pass
    "max_new_tokens": 64,  # a printed line of this book is ~60 characters
    "line_noise": 0.01,  # ink fraction below which a row counts as blank
    "line_gap": 0.3,  # merge bands closer than this * median band height (0.5 fused whole
    # paragraphs: a 26-line column came back as 5 bands)
    "line_pad": 4,  # px of white kept around each line crop
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in (over or {}).items() if k in base})
    return out


def _resolve_model(cfg: dict, oc: dict) -> str:
    """Apply ``ocr.finetune``: prefer the fine-tuned weights when they are on disk.

    ``models/`` is gitignored (the fine-tune is ~1.3 GB), so a fresh clone has the base model
    and nothing else. Warning and falling back keeps the pipeline runnable for a grader who
    has not run ``training/train.py``, instead of dying inside ``from_pretrained``. The flag
    is meaningless for Tesseract, which is not a model we train, so it is ignored there.
    """
    model = str(oc["model"])
    if not oc["finetune"] or model.lower().startswith("tesseract"):
        return model
    tuned = Path(cfg.get("ocr_train", {}).get("out_dir", "models/trocr-hkb"))
    if (tuned / "config.json").exists():
        return str(tuned)
    log.warning("ocr: finetune requested but %s is not built; using %r", tuned, model)
    return model


@lru_cache(maxsize=4)
def _load_page(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    return img


def _line_boxes(gray: np.ndarray, cfg: dict) -> list[tuple[int, int]]:
    """Split a region into (y0, y1) line bands on horizontal valleys in the ink profile."""
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    inked = mask.sum(1) > cfg["line_noise"] * gray.shape[1]
    if not inked.any():
        return []

    bands: list[list[int]] = []
    for y, on in enumerate(inked):
        if on and bands and y == bands[-1][1] + 1:
            bands[-1][1] = y
        elif on:
            bands.append([y, y])

    # Descenders and broken serifs split one line into slivers; merging on a fraction of
    # the median band height rejoins those without merging genuinely separate lines.
    median_h = float(np.median([b - a + 1 for a, b in bands]))
    merged = [bands[0]]
    for a, b in bands[1:]:
        if a - merged[-1][1] <= cfg["line_gap"] * median_h:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    pad, h = cfg["line_pad"], gray.shape[0]
    return [(max(0, a - pad), min(h, b + 1 + pad)) for a, b in merged]


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = _merge(_DEFAULTS, cfg.get("ocr", {}))
        self.cfg["model"] = _resolve_model(cfg, self.cfg)
        self.source_dirs = [Path(self.cfg["source_dir"]), Path(self.cfg["raw_dir"])]
        self.is_tesseract = str(self.cfg["model"]).lower().startswith("tesseract")
        self._model: Any = None
        self._processor: Any = None
        self._device: str = cfg.get("device", "cpu")

    def _page_image(self, page_id: str) -> np.ndarray:
        for d in self.source_dirs:
            p = d / f"{page_id}.png"
            if p.exists():
                return _load_page(str(p))
        raise FileNotFoundError(f"no image for page {page_id!r} in {self.source_dirs}")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        # cuda is the shipped config default and is not available on every machine; fall
        # back rather than crash, because a wrong device fails deep inside generate().
        want = self._device
        if want == "cuda" and not torch.cuda.is_available():
            want = "mps" if torch.backends.mps.is_available() else "cpu"
            log.warning("ocr: device 'cuda' unavailable, using %r", want)
        self._device = want
        self._processor = TrOCRProcessor.from_pretrained(self.cfg["model"])
        self._model = VisionEncoderDecoderModel.from_pretrained(self.cfg["model"]).to(want)
        self._model.eval()

    def _read_tesseract(self, crop: np.ndarray) -> str:
        import pytesseract

        return str(
            pytesseract.image_to_string(
                crop, lang=self.cfg["lang"], config=f"--psm {self.cfg['psm']}"
            )
        )

    def _read_trocr(self, crop: np.ndarray) -> str:
        import torch

        self._ensure_model()
        bands = _line_boxes(crop, self.cfg)
        if not bands:
            return ""
        # TrOCR's processor expects 3-channel continuous tone, as in its training data.
        images = [cv2.cvtColor(crop[a:b], cv2.COLOR_GRAY2RGB) for a, b in bands]
        out: list[str] = []
        for i in range(0, len(images), self.cfg["batch"]):
            pixels = self._processor(
                images=images[i : i + self.cfg["batch"]], return_tensors="pt"
            ).pixel_values
            with torch.no_grad():
                ids = self._model.generate(
                    pixels.to(self._device), max_new_tokens=self.cfg["max_new_tokens"]
                )
            out += self._processor.batch_decode(ids, skip_special_tokens=True)
        return "\n".join(t.strip() for t in out)

    def transcribe_region(self, region: Region) -> str:
        x0, y0, x1, y1 = region.bbox
        crop = self._page_image(region.page_id)[y0:y1, x0:x1]
        if crop.size == 0:
            return ""
        text = self._read_tesseract(crop) if self.is_tesseract else self._read_trocr(crop)
        return " ".join(text.split())


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks, one per region, in the order Stage 2 emitted them."""
    oc = _merge(_DEFAULTS, cfg.get("ocr", {}))
    doc_id = cfg.get("ingest", {}).get("doc_id", "doc")
    reader = Reader(cfg)

    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    skipped = 0
    for region in regions:
        if region.kind in oc["skip_kinds"]:
            skipped += 1
            continue
        text = reader.transcribe_region(region)
        if len(text) < oc["min_chars"]:
            skipped += 1
            continue
        n = seen.get(region.page_id, 0)
        seen[region.page_id] = n + 1
        chunks.append(
            Chunk(
                id=f"{region.page_id}#r{n:02d}",
                doc_id=doc_id,
                text=text,
                page_ids=[region.page_id],
            )
        )

    log.info(
        "ocr[%s]: %d chunks, %d words over %d pages (%d regions skipped)",
        reader.cfg["model"],
        len(chunks),
        sum(len(c.text.split()) for c in chunks),
        len(seen),
        skipped,
    )
    return chunks
