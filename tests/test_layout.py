"""Stage 2 layout tests — CI runs these.

Built on synthetic pages whose true reading order is known by construction, so the
assertions are about order and structure rather than about a golden output.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_agent.contracts import Page, Region
from doc_agent.vision.layout import _DEFAULTS, column_structure, detect

PAGE_H, PAGE_W = 1200, 900
MARGIN, PITCH = 60, 30


def _lines(img: np.ndarray, x0: int, x1: int, y0: int, n: int, pitch: int = PITCH) -> None:
    """Draw n text-like lines: words with spaces, and the blank rhythm real text has.

    Solid bars would be wrong on both counts — a row of 100% ink is a printed rule, which
    is exactly what the table detector looks for.
    """
    for i in range(n):
        y = y0 + i * pitch
        for x in range(x0, x1 - 10, 52):
            img[y : y + pitch // 2, x : min(x + 40, x1)] = 0


def _blank() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W), 255, np.uint8)


def _two_column() -> np.ndarray:
    img = _blank()
    _lines(img, MARGIN, 420, 200, 25)
    _lines(img, 480, PAGE_W - MARGIN, 200, 25)
    return img


def _mixed() -> np.ndarray:
    """Full-width heading and intro, then a two-column body — the hard case."""
    img = _blank()
    _lines(img, MARGIN, PAGE_W - MARGIN, 80, 1)  # running head
    _lines(img, MARGIN, PAGE_W - MARGIN, 200, 4)  # full-width intro
    _lines(img, MARGIN, 420, 420, 20)  # left column
    _lines(img, 480, PAGE_W - MARGIN, 420, 20)  # right column
    return img


def _write(tmp_path: Path, img: np.ndarray, name: str = "hkb_p0001") -> Page:
    p = tmp_path / f"{name}.png"
    cv2.imwrite(str(p), img)
    return Page(id=name, image_path=str(p), doc_id="hkb")


def _cfg(**over) -> dict:
    return {"layout": {**_DEFAULTS, **over}}


def _boxes(regions: list[Region]) -> list[tuple[int, int, int, int]]:
    return [r.bbox for r in regions]


def test_single_column_page_is_not_split_into_columns(tmp_path: Path) -> None:
    img = _blank()
    _lines(img, MARGIN, PAGE_W - MARGIN, 200, 25)
    regions = detect([_write(tmp_path, img)], _cfg())
    assert regions
    assert all((b[2] - b[0]) > 0.6 * PAGE_W for b in _boxes(regions))
    assert column_structure(regions, PAGE_W) == "single_column"


def test_two_column_page_yields_left_column_before_right(tmp_path: Path) -> None:
    regions = detect([_write(tmp_path, _two_column())], _cfg())
    assert column_structure(regions, PAGE_W) == "two_column"
    left = [b for b in _boxes(regions) if b[2] <= PAGE_W // 2]
    right = [b for b in _boxes(regions) if b[0] >= PAGE_W // 2]
    assert left and right
    # every left-column region is emitted before every right-column one
    order = [0 if b[2] <= PAGE_W // 2 else 1 for b in _boxes(regions)]
    assert order == sorted(order), "columns must not be interleaved"


def test_mixed_page_reads_banner_then_left_then_right(tmp_path: Path) -> None:
    """The failure this stage exists to fix: a full-width band must not block the gutter."""
    regions = detect([_write(tmp_path, _mixed())], _cfg())
    assert column_structure(regions, PAGE_W) == "mixed"
    boxes = _boxes(regions)
    full = [i for i, b in enumerate(boxes) if (b[2] - b[0]) > 0.75 * PAGE_W]
    left = [i for i, b in enumerate(boxes) if b[2] <= PAGE_W // 2]
    right = [i for i, b in enumerate(boxes) if b[0] >= PAGE_W // 2]
    assert full and left and right
    assert max(full) < min(left), "the full-width band must be read before the columns"
    assert max(left) < min(right), "columns must not be interleaved"


def test_regions_carry_the_page_id_and_a_usable_bbox(tmp_path: Path) -> None:
    page = _write(tmp_path, _two_column())
    img = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
    for r in detect([page], _cfg()):
        x0, y0, x1, y1 = r.bbox
        assert r.page_id == "hkb_p0001"
        assert 0 <= x0 < x1 <= img.shape[1] and 0 <= y0 < y1 <= img.shape[0]
        assert (img[y0:y1, x0:x1] < 128).any(), "a region must contain ink"


def test_scan_edge_streak_is_not_mistaken_for_a_column(tmp_path: Path) -> None:
    img = _two_column()
    img[:, 4:16] = 0  # book-edge shadow at the extreme left
    regions = detect([_write(tmp_path, img)], _cfg())
    assert all(r.bbox[0] > 20 for r in regions)
    assert column_structure(regions, PAGE_W) == "two_column"


def test_illustration_is_labelled_figure_not_text(tmp_path: Path) -> None:
    img = _blank()
    rng = np.random.default_rng(0)
    block = rng.integers(0, 2, (300, 600)) * 255  # dense, no blank-line rhythm
    img[150:450, 150:750] = block
    _lines(img, MARGIN, PAGE_W - MARGIN, 600, 10)
    kinds = {r.kind for r in detect([_write(tmp_path, img)], _cfg())}
    assert "figure" in kinds and "text" in kinds


def test_ruled_table_is_labelled_table(tmp_path: Path) -> None:
    img = _blank()
    _lines(img, MARGIN, PAGE_W - MARGIN, 220, 6)
    for y in (200, 260, 400):  # three horizontal rules
        img[y : y + 3, MARGIN : PAGE_W - MARGIN] = 0
    assert any(r.kind == "table" for r in detect([_write(tmp_path, img)], _cfg()))


def test_running_head_over_a_rule_is_a_heading_not_a_table(tmp_path: Path) -> None:
    """A single decorative rule is not a table; counting rule rows made it look like one."""
    img = _blank()
    _lines(img, MARGIN, PAGE_W - MARGIN, 80, 1)
    img[120:124, MARGIN : PAGE_W - MARGIN] = 0
    _lines(img, MARGIN, PAGE_W - MARGIN, 300, 20)
    kinds = [r.kind for r in detect([_write(tmp_path, img)], _cfg())]
    assert "table" not in kinds


def test_blank_page_yields_no_regions(tmp_path: Path) -> None:
    assert detect([_write(tmp_path, _blank())], _cfg()) == []


def test_unreadable_image_fails_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="unreadable image"):
        detect([Page(id="hkb_p0001", image_path=str(bad), doc_id="hkb")], _cfg())


def test_pages_are_kept_in_order(tmp_path: Path) -> None:
    pages = [_write(tmp_path, _two_column(), f"hkb_p{i:04d}") for i in (1, 2, 3)]
    seen = [r.page_id for r in detect(pages, _cfg())]
    assert seen == sorted(seen)
