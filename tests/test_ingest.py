"""Stage 1 ingest tests — CI runs these."""

from pathlib import Path

import pytest
import yaml

from doc_agent.ingest.loader import load_pages

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"


def _cfg(raw_dir: Path) -> dict:
    return {"ingest": {"raw_dir": str(raw_dir), "doc_id": "hkb", "page_glob": "*.png"}}


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_bytes(b"")


def test_page_ids_are_zero_padded_and_numerically_ordered(tmp_path: Path) -> None:
    # written out of order, and 10 must not sort before 2
    _touch(tmp_path, "hkb-10.png", "hkb-2.png", "hkb-1.png")
    pages = load_pages(_cfg(tmp_path))
    assert [p.id for p in pages] == ["hkb_p0001", "hkb_p0002", "hkb_p0010"]


def test_page_carries_doc_id_and_readable_path(tmp_path: Path) -> None:
    _touch(tmp_path, "hkb-007.png")
    (page,) = load_pages(_cfg(tmp_path))
    assert page.doc_id == "hkb"
    assert Path(page.image_path).is_file()


def test_non_matching_files_are_ignored(tmp_path: Path) -> None:
    _touch(tmp_path, "hkb-1.png", "notes.txt", "hkb-2.jpg")
    assert [p.id for p in load_pages(_cfg(tmp_path))] == ["hkb_p0001"]


def test_missing_dir_points_at_get_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="get_data.sh"):
        load_pages(_cfg(tmp_path / "nope"))


def test_empty_dir_points_at_get_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="get_data.sh"):
        load_pages(_cfg(tmp_path))


def test_duplicate_page_numbers_are_rejected(tmp_path: Path) -> None:
    _touch(tmp_path, "a-5.png", "b-5.png")
    with pytest.raises(ValueError, match="duplicate"):
        load_pages(_cfg(tmp_path))


@pytest.mark.skipif(
    not RAW.is_dir() or not any(RAW.glob("*.png")),
    reason="corpus not built — run scripts/get_data.sh",
)
def test_real_corpus_matches_a1_declaration() -> None:
    """The corpus on disk must match what configs/task.yaml declares (A1 Section 2)."""
    task = yaml.safe_load((REPO / "configs" / "task.yaml").read_text())
    cfg = yaml.safe_load((REPO / "configs" / "config.yaml").read_text())
    cfg["ingest"]["raw_dir"] = str(RAW)

    pages = load_pages(cfg)
    assert len(pages) == 624, "The Home-Keeping Book (1918) has 624 scanned pages"
    assert len(pages) >= task["corpus"]["min_pages"]
    assert pages[0].id == "hkb_p0001"
    assert pages[-1].id == "hkb_p0624"
    # PDF page 350 = printed folio 250, the two-column page A1 uses as its worst case
    assert pages[349].id == "hkb_p0350"
