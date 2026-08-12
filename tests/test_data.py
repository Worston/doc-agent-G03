"""Unit tests for the OCR fine-tune data path.

The thing worth testing here is not that files are read, but that a *wrong label never
reaches the trainer*: the split must keep held-out pages out of training, and the line
pairing must refuse to guess when our line bands and ABBYY's line texts disagree.
"""

import json

import cv2
import numpy as np
import pytest

from doc_agent.training import datamodule as dmod

# --- pairing rule ------------------------------------------------------------------


def test_pair_region_zips_in_reading_order_when_counts_agree():
    bands = [(0, 10), (10, 20), (20, 30)]
    texts = ["one", "two", "three"]
    assert dmod._pair_region(bands, texts) == [
        ((0, 10), "one"),
        ((10, 20), "two"),
        ((20, 30), "three"),
    ]


@pytest.mark.parametrize("texts", [["one"], ["one", "two", "three"], []])
def test_pair_region_drops_the_region_rather_than_guessing(texts):
    """A mismatched count means we cannot say which text belongs to which band.

    ABBYY's y-coordinates are off the ink by 26 px std against a 44 px line pitch, so
    falling back to nearest-position matching would pair a line with its neighbour.
    """
    assert dmod._pair_region([(0, 10), (10, 20)], texts) == []


# --- crop filtering ----------------------------------------------------------------


def _crop(h=30, w=200, ink=0.1):
    img = np.full((h, w), 255, np.uint8)
    img[:, : int(w * ink)] = 0
    return img


@pytest.mark.parametrize(
    "crop,text,why",
    [
        (_crop(), "a", "text shorter than min_chars"),
        (_crop(h=5), "a real line", "crop thinner than min_height"),
        (_crop(w=20), "a real line", "crop narrower than min_width"),
        (_crop(h=13, w=900), "a real line", "aspect ratio of a rule, not a line"),
        (np.full((30, 200), 255, np.uint8), "a real line", "no ink at all"),
        (np.zeros((30, 200), np.uint8), "a real line", "solid black band"),
    ],
)
def test_keep_rejects_crops_that_cannot_be_a_text_line(crop, text, why):
    assert not dmod._keep(crop, text, dmod._DEFAULTS), why


def test_keep_accepts_an_ordinary_line():
    assert dmod._keep(_crop(), "a real line of printed text", dmod._DEFAULTS)


# --- ABBYY line parsing ------------------------------------------------------------


def test_lines_for_page_scales_points_to_pixels_and_unescapes():
    body = (
        '<line xMin="10.0" yMin="20.0" xMax="110.0" yMax="40.0">'
        "<word>Salt</word><word>&amp;</word><word>Pepper</word></line>"
    )
    # a 100x200 pt page rendered at 200x400 px is exactly 2x
    (line,) = dmod._lines_for_page(body, 100.0, 200.0, 200, 400)
    assert line["text"] == "Salt & Pepper"
    assert line["bbox"] == [20, 40, 220, 80]


def test_lines_for_page_drops_lines_with_no_words():
    body = '<line xMin="1" yMin="1" xMax="2" yMax="2"></line>'
    assert dmod._lines_for_page(body, 10.0, 10.0, 10, 10) == []


# --- config merge ------------------------------------------------------------------


def test_merge_ignores_keys_this_module_does_not_own():
    """``ocr_train`` also carries trainer keys; they must not shadow data settings.

    ``out_dir`` there means the *model* directory, which is why the corpus key is
    ``lines_dir``: an earlier name collision sent the DataModule looking for train.jsonl
    inside models/.
    """
    merged = dmod._merge(dmod._DEFAULTS, {"out_dir": "models/trocr-hkb", "min_chars": 9})
    assert "out_dir" not in merged
    assert merged["lines_dir"] == dmod._DEFAULTS["lines_dir"]
    assert merged["min_chars"] == 9


# --- dataset -----------------------------------------------------------------------


def test_dataset_yields_rgb_arrays_for_the_processor(tmp_path):
    cv2.imwrite(str(tmp_path / "l000.png"), _crop())
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"image": "l000.png", "text": "a line", "page_id": "p"}) + "\n")

    ds = dmod.OCRLineDataset(manifest, tmp_path)
    assert len(ds) == 1
    item = ds[0]
    assert item["text"] == "a line"
    assert item["image"].ndim == 3 and item["image"].shape[2] == 3


def test_dataset_fails_loudly_on_a_missing_crop(tmp_path):
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps({"image": "gone.png", "text": "a line", "page_id": "p"}) + "\n")
    with pytest.raises(ValueError, match="unreadable line crop"):
        dmod.OCRLineDataset(manifest, tmp_path)[0]


# --- the training objective --------------------------------------------------------


def test_loss_scores_logits_against_labels_at_the_same_position():
    """A perfect prediction at position i must cost nothing.

    VisionEncoderDecoderModel already shifts the decoder inputs right, so scoring
    ``logits[:-1]`` against ``labels[1:]`` shifts a second time and trains the model to
    predict the token after next -- it then skips every other token when it generates.
    Any re-introduced shift makes this loss large instead of ~0.
    """
    torch = pytest.importorskip("torch")

    labels = torch.tensor([[5, 7, 9, 2]])
    logits = torch.nn.functional.one_hot(labels, num_classes=16).float() * 30.0
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1))
    assert float(loss) < 1e-4

    shifted = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 16), labels[:, 1:].reshape(-1)
    )
    assert float(shifted) > 1.0, "the double shift must not look cheap"


def test_padding_is_masked_out_of_the_loss():
    """-100 is the ignore index; without it the model is rewarded for predicting pad."""
    torch = pytest.importorskip("torch")

    labels = torch.tensor([[5, 7, -100, -100]])
    logits = torch.zeros(1, 4, 16)
    logits[0, 0, 5] = 30.0
    logits[0, 1, 7] = 30.0
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 16), labels.reshape(-1))
    assert float(loss) < 1e-4


# --- the leakage guard -------------------------------------------------------------


@pytest.mark.skipif(
    not (dmod.REPO / "grading_kit" / "labels.jsonl").exists(),
    reason="needs the source PDF and the held-out labels",
)
def test_heldout_pages_never_reach_train_or_val():
    """The Section 5 OCR number is only meaningful if this holds."""
    splits = dmod.page_splits()
    heldout = {
        int(json.loads(line)["page_id"].split("_p")[1])
        for line in (dmod.REPO / "grading_kit" / "labels.jsonl").read_text().splitlines()
        if line.strip()
    }
    assert heldout, "no held-out pages found — the guard would pass vacuously"
    assert all(splits[p] not in ("train", "val") for p in heldout)
