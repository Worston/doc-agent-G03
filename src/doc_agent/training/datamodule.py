"""Training data for the OCR fine-tune: (line image, line text) pairs.

TrOCR is a *line* recogniser, so it cannot be trained on page transcriptions. The labels
come from the Internet Archive's own ABBYY text layer, which ``pdftotext -bbox-layout``
emits as ``<line>`` elements carrying both the text and a bounding box — that is, already
aligned line pairs, for the whole book, at no annotation cost.

Those labels are not perfect, and their quality was measured rather than assumed. Scored
against the 40 human-verified pages the raw text layer gives CER 0.185, which looks
disqualifying until you notice its word count is right (27,091 against 26,397): the error
is reading *order*, not glyphs, and the same failure Stage 2 exists to fix. Re-ordering
ABBYY's lines with our own layout regions gives:

    layout           n     raw     reordered
    single_column    6   0.0367      0.0531
    two_column      18   0.1133      0.0393
    mixed           16   0.3213      0.0548
    all             40   0.1850      0.0476

So the text layer is ~95% character-accurate per line, and per-line order is irrelevant
when each line is its own training example. Training on it is therefore **distillation
from a noisy teacher**, not learning from ground truth, and it must be described that way:
the only ground truth in this project is the 40 pages a human checked.

Which makes the split the load-bearing part of this module. The 40 held-out pages and
every page of a TEST section are excluded from training, using ``section_map`` imported
from ``grading_kit/heldout_pages/select_heldout.py`` — the *same* code that drew the
held-out slice, so the guard cannot drift from the thing it is guarding. VAL sections
become the validation set. Without that, the fine-tuned model would be evaluated on pages
whose ABBYY labels it had already been trained on, and the Section 5 OCR number would be
meaningless.
"""

from __future__ import annotations

import html
import importlib.util
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import lightning as pl
import numpy as np
import yaml
from torch.utils.data import DataLoader, Dataset

from ..contracts import Page
from ..logging_conf import get_logger
from ..vision import layout, ocr

log = get_logger(__name__)

REPO = Path(__file__).resolve().parents[3]

_LINE = re.compile(
    r'<line xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</line>', re.S
)
_WORD = re.compile(r"<word[^>]*>(.*?)</word>", re.S)
_PAGE = re.compile(r'<page width="([\d.]+)" height="([\d.]+)">(.*?)</page>', re.S)

_DEFAULTS: dict = {
    "lines_dir": "data/interim/ocr_lines",
    "source_dir": "data/interim",
    "min_chars": 4,  # "a." teaches nothing and is usually a stray marginal number
    "min_height": 12,  # px; below this the crop is a rule fragment, not a line
    "min_width": 40,
    "max_aspect": 60.0,  # width/height; a running head rule can slip through as a "line"
    "min_ink": 0.01,  # crop must actually contain ink...
    "max_ink": 0.60,  # ...and not be a solid black band (bleed-through, scan edge)
    "pad": 4,
    "max_train_pages": 0,  # 0 = all
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in (over or {}).items() if k in base})
    return out


def _selector() -> Any:
    """Import the held-out selector so the split rule has exactly one implementation."""
    path = REPO / "grading_kit" / "heldout_pages" / "select_heldout.py"
    spec = importlib.util.spec_from_file_location("select_heldout", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the held-out selector at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_splits() -> dict[int, str]:
    """PDF page number -> 'train' | 'val' | 'test' | 'exclude', by SECTION as A1 committed."""
    sel = _selector()
    task = yaml.safe_load((REPO / "configs" / "task.yaml").read_text())
    pdf = REPO / task["eda"]["source_pdf"]

    raw = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout.split("\f")
    stats = sel.page_stats(sel._pdftotext("-bbox-layout"))
    unit = sel.section_map(raw, stats)

    heldout = {
        int(json.loads(line)["page_id"].split("_p")[1])
        for line in (REPO / "grading_kit" / "labels.jsonl").read_text().splitlines()
        if line.strip()
    }

    out: dict[int, str] = {}
    for page in range(1, len(raw) + 1):
        section = unit.get(page)
        if page in heldout or page in sel.EXCL or section in sel.TEST_SECTIONS:
            # held-out pages and whole TEST sections never reach the trainer
            out[page] = "test" if section in sel.TEST_SECTIONS or page in heldout else "exclude"
        elif section in sel.VAL_SECTIONS:
            out[page] = "val"
        else:
            out[page] = "train"
    return out


def _lines_for_page(body: str, pw: float, ph: float, w: int, h: int) -> list[dict]:
    sx, sy = w / pw, h / ph
    out = []
    for x0, y0, x1, y1, inner in _LINE.findall(body):
        text = re.sub(r"\s+", " ", html.unescape(" ".join(_WORD.findall(inner)))).strip()
        if text:
            out.append(
                {
                    "text": text,
                    "bbox": [
                        int(float(x0) * sx),
                        int(float(y0) * sy),
                        int(float(x1) * sx),
                        int(float(y1) * sy),
                    ],
                }
            )
    return out


def _pair_region(
    bands: list[tuple[int, int]], texts: list[str]
) -> list[tuple[tuple[int, int], str]]:
    """Zip our line bands to ABBYY's line texts, in order, only when the counts agree.

    ABBYY's y-coordinates do not land on the ink: measured against the actual ink bands the
    offset has median -5 px but standard deviation 26 px, against a line pitch of ~44 px,
    so matching a text to a band by position would silently pair a line with its neighbour.
    Within one layout region both sequences are in reading order, so when they are the same
    length the k-th text belongs to the k-th band. When they are not, the region is dropped
    rather than guessed at — there are 27k pairs and a wrong label is worse than a lost one.
    """
    return list(zip(bands, texts, strict=True)) if len(bands) == len(texts) else []


def _keep(crop: np.ndarray, text: str, cfg: dict) -> bool:
    h, w = crop.shape[:2]
    if len(text) < cfg["min_chars"] or h < cfg["min_height"] or w < cfg["min_width"]:
        return False
    if w / max(h, 1) > cfg["max_aspect"]:
        return False
    ink = float((crop < 128).mean())
    return cfg["min_ink"] <= ink <= cfg["max_ink"]


def build_line_corpus(cfg: dict) -> dict[str, int]:
    """Write line crops + a manifest per split. Returns the pair count per split."""
    tc = _merge(_DEFAULTS, cfg.get("ocr_train", {}))
    task = yaml.safe_load((REPO / "configs" / "task.yaml").read_text())
    pdf = REPO / task["eda"]["source_pdf"]

    out_dir = REPO / tc["lines_dir"]
    src_dir = REPO / tc["source_dir"]
    for split in ("train", "val"):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    splits = page_splits()
    xml = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"], capture_output=True, text=True, check=True
    ).stdout
    pages = _PAGE.findall(xml)

    manifests: dict[str, list[dict]] = {"train": [], "val": []}
    dropped = 0
    for page_no, (pw, ph, body) in enumerate(pages, 1):
        split = splits.get(page_no, "train")
        if split not in manifests:
            continue
        if tc["max_train_pages"] and len(manifests[split]) > tc["max_train_pages"]:
            continue
        page_id = f"hkb_p{page_no:04d}"
        img_path = src_dir / f"{page_id}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        abbyy = _lines_for_page(body, float(pw), float(ph), w, h)
        page = Page(id=page_id, image_path=str(img_path), doc_id="hkb")

        idx = 0
        for region in layout.detect([page], cfg):
            if region.kind == "figure":
                continue
            rx0, ry0, rx1, ry1 = region.bbox
            crop_region = img[ry0:ry1, rx0:rx1]
            bands = [(ry0 + a, ry0 + b) for a, b in ocr._line_boxes(crop_region, ocr._DEFAULTS)]
            texts = [
                ln["text"]
                for ln in abbyy
                if rx0 <= (ln["bbox"][0] + ln["bbox"][2]) / 2 <= rx1
                and ry0 <= (ln["bbox"][1] + ln["bbox"][3]) / 2 <= ry1
            ]
            pairs = _pair_region(bands, texts)
            if not pairs:
                dropped += len(bands)
                continue
            for (by0, by1), text in pairs:
                crop = img[by0:by1, rx0:rx1]
                if crop.size == 0 or not _keep(crop, text, tc):
                    dropped += 1
                    continue
                rel = f"{split}/{page_id}_l{idx:03d}.png"
                cv2.imwrite(str(out_dir / rel), crop)
                manifests[split].append({"image": rel, "text": text, "page_id": page_id})
                idx += 1

    counts = {}
    for split, rows in manifests.items():
        with (out_dir / f"{split}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        counts[split] = len(rows)

    n_pages = {s: sum(1 for v in splits.values() if v == s) for s in ("train", "val", "test")}
    log.info(
        "ocr line corpus: train %d / val %d pairs (%d dropped) from pages train %d / val %d, "
        "test+heldout %d excluded",
        counts["train"],
        counts["val"],
        dropped,
        n_pages["train"],
        n_pages["val"],
        n_pages["test"],
    )
    return counts


class OCRLineDataset(Dataset):
    """(line image, text) pairs. Yields RGB arrays, which is what TrOCRProcessor expects."""

    def __init__(self, manifest: str | Path, root: str | Path) -> None:
        self.root = Path(root)
        self.rows = [
            json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        img = cv2.imread(str(self.root / row["image"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"unreadable line crop: {row['image']}")
        return {"image": cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), "text": row["text"]}


class DocDataModule(pl.LightningDataModule):
    """Split by DOCUMENT SECTION, not by line: see ``page_splits``.

    Splitting lines at random would put lines from a held-out page into training, which is
    the leak this project's whole evaluation story depends on not happening.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = _merge(_DEFAULTS, cfg.get("ocr_train", {}))
        self.batch_size = int(cfg.get("ocr_train", {}).get("batch_size", 8))
        self.workers = int(cfg.get("ocr_train", {}).get("workers", 0))
        self.root = REPO / self.cfg["lines_dir"]
        self.train: OCRLineDataset | None = None
        self.val: OCRLineDataset | None = None

    def prepare_data(self) -> None:
        if not (self.root / "train.jsonl").exists():
            build_line_corpus({"ocr_train": self.cfg})

    def setup(self, stage: str | None = None) -> None:
        self.train = OCRLineDataset(self.root / "train.jsonl", self.root)
        self.val = OCRLineDataset(self.root / "val.jsonl", self.root)

    def _loader(self, ds: OCRLineDataset | None, shuffle: bool) -> DataLoader:
        assert ds is not None, "call setup() first"
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            collate_fn=lambda b: {
                "image": [x["image"] for x in b],
                "text": [x["text"] for x in b],
            },
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train, True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val, False)
