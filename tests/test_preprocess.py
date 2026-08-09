"""Stage 1 preprocessing tests — CI runs these.

Skew recovery is checked against a *known* injected angle on a synthetic page, so the
assertion is on measured error rather than on a golden output file.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from doc_agent.contracts import Page
from doc_agent.ingest.preprocess import (
    _DEFAULTS,
    _despeckle,
    _merge,
    _rotate,
    estimate_skew,
    preprocess_image,
    run,
)


def _text_page(h: int = 900, w: int = 700) -> np.ndarray:
    """Paper-toned page with evenly spaced text lines, plus paper grain.

    Continuous-tone on purpose: a perfectly bilevel synthetic page is separable by any
    global threshold and would let a broken binarizer pass.
    """
    img = np.full((h, w), 235, np.uint8)
    for y in range(60, h - 60, 40):
        img[y : y + 14, 60 : w - 60] = 90  # letterpress grey, not pure black
    grain = np.random.default_rng(0).normal(0, 6, img.shape)
    return np.clip(img + grain, 0, 255).astype(np.uint8)


def _ink_fraction(mask: np.ndarray) -> float:
    return float((mask == 0).mean())


def _cfg(**over) -> dict:
    return _merge(_DEFAULTS, over)


@pytest.mark.parametrize("truth", [-1.2, -0.4, 0.0, 0.35, 1.1])
def test_skew_is_recovered_within_the_search_resolution(truth: float) -> None:
    skewed = _rotate(_text_page(), truth, 235)
    assert abs(estimate_skew(skewed, _DEFAULTS["deskew"]) + truth) <= _DEFAULTS["deskew"]["fine_step"] * 2


def test_blank_page_reports_no_skew() -> None:
    # No ink means no profile to sharpen. Without the guard every angle ties on score and
    # array order wins, rotating a blank page by the full search limit.
    assert estimate_skew(np.full((400, 400), 255, np.uint8), _DEFAULTS["deskew"]) == 0.0


def test_estimate_never_leaves_the_search_window() -> None:
    """The fine sweep brackets the coarse winner, so it can overshoot max_angle unclipped."""
    cfg = dict(_DEFAULTS["deskew"], max_angle=1.0)
    page = _rotate(_text_page(), 2.5, 235)  # skewed further than the window allows
    assert abs(estimate_skew(page, cfg)) <= cfg["max_angle"]


def test_despeckle_drops_grit_and_keeps_punctuation() -> None:
    img = np.full((200, 200), 255, np.uint8)
    img[20:23, 20:23] = 0  # 9 px speck
    img[100:110, 100:110] = 0  # 100 px full stop
    out = _despeckle(img, min_area=12)
    assert out[20:23, 20:23].min() == 255
    assert out[100:110, 100:110].max() == 0


def test_binarize_none_keeps_greyscale_and_sauvola_makes_two_levels() -> None:
    page = _text_page()
    grey, _ = preprocess_image(page, _cfg(binarize={"method": "none"}))
    binary, _ = preprocess_image(page, _cfg(binarize={"method": "sauvola"}))
    assert len(np.unique(grey)) > 2
    assert set(np.unique(binary)) <= {0, 255}


def test_sauvola_survives_a_brightness_gradient_that_defeats_otsu() -> None:
    """The gutter margin is darker than the fore-edge; a global threshold floods it."""
    page = _text_page()
    truth = float((page < 160).mean())
    gradient = np.linspace(0.45, 1.0, page.shape[1], dtype=np.float32)[None, :]
    shaded = np.clip(page * gradient, 0, 255).astype(np.uint8)

    # Sauvola tracks the local background, so it recovers close to the true ink fraction...
    sauvola = _ink_fraction(preprocess_image(shaded, _cfg(binarize={"method": "sauvola"}))[0])
    assert sauvola == pytest.approx(truth, abs=0.02)
    # ...while one global threshold cannot serve both sides and floods the dark margin.
    assert _ink_fraction(preprocess_image(shaded, _cfg(binarize={"method": "otsu"}))[0]) > truth + 0.15


def test_unknown_binarize_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown binarize.method"):
        preprocess_image(_text_page(), _cfg(binarize={"method": "quantum"}))


def _pages(tmp_path: Path, n: int = 3) -> tuple[list[Page], dict]:
    raw = tmp_path / "raw"
    raw.mkdir()
    pages = []
    for i in range(1, n + 1):
        p = raw / f"hkb-{i}.png"
        cv2.imwrite(str(p), _rotate(_text_page(), 0.8, 235))
        pages.append(Page(id=f"hkb_p{i:04d}", image_path=str(p), doc_id="hkb"))
    return pages, {"preprocess": {"out_dir": str(tmp_path / "interim"), "workers": 1}}


def test_run_preserves_ids_and_repoints_pages_at_written_files(tmp_path: Path) -> None:
    pages, cfg = _pages(tmp_path)
    out = run(pages, cfg)
    assert [p.id for p in out] == [p.id for p in pages]
    assert [p.doc_id for p in out] == ["hkb"] * len(pages)
    for src, dst in zip(pages, out):
        assert dst.image_path != src.image_path
        assert Path(dst.image_path).is_file()
        assert cv2.imread(dst.image_path, cv2.IMREAD_GRAYSCALE) is not None


def test_second_run_is_cached_but_a_config_change_invalidates_it(tmp_path: Path) -> None:
    pages, cfg = _pages(tmp_path, n=2)
    first = [Path(p.image_path) for p in run(pages, cfg)]
    stamps = [p.stat().st_mtime_ns for p in first]

    run(pages, cfg)
    assert [p.stat().st_mtime_ns for p in first] == stamps, "cache hit must not rewrite files"

    cfg["preprocess"]["binarize"] = {"method": "otsu"}
    run(pages, cfg)
    assert [p.stat().st_mtime_ns for p in first] != stamps, "changed config must invalidate cache"


def test_disabled_is_a_pass_through(tmp_path: Path) -> None:
    pages, cfg = _pages(tmp_path, n=1)
    cfg["preprocess"]["enabled"] = False
    assert run(pages, cfg) == pages


def test_unreadable_image_fails_loudly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not a png")
    with pytest.raises(ValueError, match="unreadable image"):
        run([Page(id="hkb_p0001", image_path=str(bad), doc_id="hkb")],
            {"preprocess": {"out_dir": str(tmp_path / "out"), "workers": 1}})
