"""Stage 3 OCR tests — CI runs these.

The recogniser itself is not under test here: it is a third-party model, and asserting on
its output would make CI a weather report. What is under test is everything this module
owns — line segmentation, region-to-chunk mapping, ordering, and the contract fields.
"""

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_agent.contracts import Region
from doc_agent.vision.ocr import _DEFAULTS, Reader, _line_boxes, transcribe

PITCH, LINE_H = 40, 18


def _page(tmp_path: Path, n_lines: int = 6, name: str = "hkb_p0001") -> Path:
    """A page of text-like lines with a known line count and pitch."""
    img = np.full((PITCH * n_lines + 80, 700), 255, np.uint8)
    for i in range(n_lines):
        y = 40 + i * PITCH
        for x in range(40, 640, 60):
            img[y : y + LINE_H, x : x + 45] = 0
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), img)
    return p


def _cfg(tmp_path: Path, **over) -> dict:
    return {"ingest": {"doc_id": "hkb"}, "ocr": {"source_dir": str(tmp_path), **over}}


def _region(kind: str = "text", bbox: tuple = (0, 0, 700, 320), pid: str = "hkb_p0001") -> Region:
    return Region(page_id=pid, bbox=bbox, kind=kind)


# --- line segmentation -------------------------------------------------------


def test_line_boxes_finds_one_band_per_printed_line(tmp_path: Path) -> None:
    img = cv2.imread(str(_page(tmp_path, 6)), cv2.IMREAD_GRAYSCALE)
    assert len(_line_boxes(img, _DEFAULTS)) == 6


def test_line_bands_are_ordered_disjoint_and_contain_ink(tmp_path: Path) -> None:
    img = cv2.imread(str(_page(tmp_path, 6)), cv2.IMREAD_GRAYSCALE)
    bands = _line_boxes(img, _DEFAULTS)
    for (a, b), (c, _) in zip(bands, bands[1:], strict=False):
        assert a < b <= c, "bands must be ordered and non-overlapping"
    assert all((img[a:b] < 128).any() for a, b in bands)


def test_a_generous_line_gap_fuses_lines_which_is_why_the_default_is_small(
    tmp_path: Path,
) -> None:
    """Regression: line_gap 0.5 collapsed a 26-line column into 5 bands, and TrOCR then
    emitted one hallucinated line per paragraph instead of one line per printed line."""
    img = cv2.imread(str(_page(tmp_path, 6)), cv2.IMREAD_GRAYSCALE)
    fused = _line_boxes(img, {**_DEFAULTS, "line_gap": 2.0})
    assert len(fused) < len(_line_boxes(img, _DEFAULTS))


def test_blank_region_yields_no_lines() -> None:
    assert _line_boxes(np.full((200, 400), 255, np.uint8), _DEFAULTS) == []


# --- region -> chunk mapping -------------------------------------------------


def test_chunks_preserve_region_order_and_carry_contract_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _page(tmp_path)
    monkeypatch.setattr(Reader, "transcribe_region", lambda self, r: f"text at {r.bbox[1]}")
    regions = [_region(bbox=(0, y, 700, y + 100)) for y in (0, 100, 200)]
    chunks = transcribe(regions, _cfg(tmp_path))

    assert [c.text for c in chunks] == ["text at 0", "text at 100", "text at 200"]
    assert [c.id for c in chunks] == ["hkb_p0001#r00", "hkb_p0001#r01", "hkb_p0001#r02"]
    assert all(c.doc_id == "hkb" and c.page_ids == ["hkb_p0001"] for c in chunks)


def test_chunk_ids_are_unique_across_pages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Reader, "transcribe_region", lambda self, r: "words here")
    regions = [_region(pid=p) for p in ("hkb_p0001", "hkb_p0002") for _ in range(3)]
    ids = [c.id for c in transcribe(regions, _cfg(tmp_path))]
    assert len(set(ids)) == len(ids) == 6


def test_figures_are_skipped_and_other_kinds_are_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Reader, "transcribe_region", lambda self, r: "some words")
    regions = [_region("text"), _region("figure"), _region("heading"), _region("table")]
    assert len(transcribe(regions, _cfg(tmp_path))) == 3


def test_regions_recognising_to_noise_are_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        Reader, "transcribe_region", lambda self, r: "." if r.kind == "table" else "real text"
    )
    chunks = transcribe([_region("table"), _region("text")], _cfg(tmp_path))
    assert [c.text for c in chunks] == ["real text"]


def test_no_regions_yields_no_chunks(tmp_path: Path) -> None:
    assert transcribe([], _cfg(tmp_path)) == []


# --- image resolution --------------------------------------------------------


def test_reader_prefers_the_preprocessed_image_over_raw(tmp_path: Path) -> None:
    """Stage 2 measured its bboxes on the interim image, so reading raw would misalign."""
    interim, raw = tmp_path / "interim", tmp_path / "raw"
    interim.mkdir()
    raw.mkdir()
    cv2.imwrite(str(interim / "hkb_p0001.png"), np.full((50, 50), 200, np.uint8))
    cv2.imwrite(str(raw / "hkb_p0001.png"), np.full((50, 50), 100, np.uint8))
    reader = Reader({"ocr": {"source_dir": str(interim), "raw_dir": str(raw)}})
    assert reader._page_image("hkb_p0001")[0, 0] == 200


def test_raw_is_used_when_preprocessing_produced_nothing(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    cv2.imwrite(str(raw / "hkb_p0001.png"), np.full((50, 50), 100, np.uint8))
    reader = Reader({"ocr": {"source_dir": str(tmp_path / "missing"), "raw_dir": str(raw)}})
    assert reader._page_image("hkb_p0001")[0, 0] == 100


def test_missing_page_image_fails_loudly(tmp_path: Path) -> None:
    reader = Reader({"ocr": {"source_dir": str(tmp_path), "raw_dir": str(tmp_path)}})
    with pytest.raises(FileNotFoundError, match="hkb_p0001"):
        reader._page_image("hkb_p0001")


def test_unreadable_image_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "hkb_p0001.png").write_bytes(b"not a png")
    reader = Reader({"ocr": {"source_dir": str(tmp_path), "raw_dir": str(tmp_path)}})
    with pytest.raises(ValueError, match="unreadable image"):
        reader._page_image("hkb_p0001")


# --- which weights get loaded ------------------------------------------------


def test_finetune_uses_the_trained_weights_when_they_exist(tmp_path: Path) -> None:
    tuned = tmp_path / "models" / "trocr-hkb"
    tuned.mkdir(parents=True)
    (tuned / "config.json").write_text("{}")
    cfg = _cfg(tmp_path, model="microsoft/trocr-base-printed", finetune=True)
    cfg["ocr_train"] = {"out_dir": str(tuned)}
    assert Reader(cfg).cfg["model"] == str(tuned)


def test_finetune_falls_back_to_the_base_model_when_not_built(tmp_path: Path) -> None:
    """models/ is gitignored, so a fresh clone must degrade rather than crash."""
    cfg = _cfg(tmp_path, model="microsoft/trocr-base-printed", finetune=True)
    cfg["ocr_train"] = {"out_dir": str(tmp_path / "nothing-here")}
    assert Reader(cfg).cfg["model"] == "microsoft/trocr-base-printed"


def test_finetune_never_overrides_an_explicit_tesseract_choice(tmp_path: Path) -> None:
    tuned = tmp_path / "models" / "trocr-hkb"
    tuned.mkdir(parents=True)
    (tuned / "config.json").write_text("{}")
    cfg = _cfg(tmp_path, model="tesseract", finetune=True)
    cfg["ocr_train"] = {"out_dir": str(tuned)}
    reader = Reader(cfg)
    assert reader.cfg["model"] == "tesseract" and reader.is_tesseract


# --- the real recogniser, when the machine has one ---------------------------


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_tesseract_backend_reads_rendered_words(tmp_path: Path) -> None:
    img = np.full((120, 700), 255, np.uint8)
    cv2.putText(img, "HOME KEEPING", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
    cv2.imwrite(str(tmp_path / "hkb_p0001.png"), img)
    chunks = transcribe([_region(bbox=(0, 0, 700, 120))], _cfg(tmp_path, model="tesseract"))
    assert "KEEPING" in chunks[0].text.upper()
