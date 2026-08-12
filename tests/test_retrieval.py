"""Stage 4 tests — CI runs these.

Chunking is where a citation is either preserved or lost, so what is tested is the
bookkeeping this module owns: that windows respect the wordpiece budget, that they
overlap and always advance, that a window straddling a page break remembers both pages,
and that the text is never round-tripped through the tokenizer.
"""

import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk as ch

CFG = {"index": {"chunk_tokens": 12, "overlap": 4}, "embed": {"model": "stub"}}
BUDGET = CFG["index"]["chunk_tokens"] - ch._SPECIALS  # 10 words when 1 piece == 1 word


@pytest.fixture(autouse=True)
def one_piece_per_word(monkeypatch):
    """Make a wordpiece a word, so the arithmetic in these tests is readable."""

    class _Stub:
        def __call__(self, words, add_special_tokens=False):
            return {"input_ids": [[0] for _ in words]}

    monkeypatch.setattr(ch, "_tokenizer", lambda name: _Stub())


def _region(text: str, page: str = "hkb_p0001", n: int = 0) -> Chunk:
    return Chunk(id=f"{page}#r{n:02d}", doc_id="hkb", text=text, page_ids=[page])


def _words(n: int, first: int = 0) -> str:
    return " ".join(f"w{i:03d}" for i in range(first, first + n))


# --- windowing ---------------------------------------------------------------------


def test_windows_never_exceed_the_budget():
    spans = ch._windows([1] * 100, 10, 3)
    assert all(b - a <= 10 for a, b in spans)


def test_windows_overlap_and_cover_every_word():
    spans = ch._windows([1] * 100, 10, 3)
    assert spans[0][0] == 0 and spans[-1][1] == 100
    for (a, b), (c, _) in zip(spans, spans[1:], strict=False):
        assert a < c < b, "each window must start inside the previous one and advance"


def test_a_word_longer_than_the_budget_still_gets_a_window():
    """Otherwise the loop cannot advance past it and the document is lost from here on."""
    assert ch._windows([3, 99, 3], 10, 2) == [(0, 1), (1, 2), (2, 3)]


def test_zero_overlap_partitions_without_repeating():
    spans = ch._windows([1] * 30, 10, 0)
    assert spans == [(0, 10), (10, 20), (20, 30)]


def test_no_words_yields_no_windows():
    assert ch._windows([], 10, 3) == []


# --- region -> window mapping ------------------------------------------------------


def test_short_consecutive_regions_are_fused_into_one_window():
    """39 of 96 real regions are under 32 tokens; alone they are too small to retrieve."""
    regions = [_region("Tomato Soup", n=0), _region("Serves six", n=1), _region("Boil it", n=2)]
    (out,) = ch.split(regions, CFG)
    assert out.text == "Tomato Soup Serves six Boil it"


def test_a_long_region_is_cut_into_overlapping_windows():
    out = ch.split([_region(_words(25))], CFG)
    assert len(out) == 4  # budget 10, overlap 4 -> stride 6
    assert out[0].text.split()[:1] == ["w000"]
    assert out[-1].text.split()[-1] == "w024"
    assert set(out[0].text.split()) & set(out[1].text.split()), "windows must share context"


def test_window_ids_are_unique_and_sequential():
    ids = [c.id for c in ch.split([_region(_words(60))], CFG)]
    assert ids[:2] == ["hkb#c00000", "hkb#c00001"]
    assert len(set(ids)) == len(ids)


def test_documents_are_chunked_independently():
    a = _region(_words(5))
    b = Chunk(id="x#r00", doc_id="other", text=_words(5), page_ids=["x_p0001"])
    out = ch.split([a, b], CFG)
    assert {c.doc_id for c in out} == {"hkb", "other"}
    assert all(len({w for w in c.text.split()}) == 5 for c in out), "no bleed between docs"


def test_no_chunks_yields_no_chunks():
    assert ch.split([], CFG) == []


# --- citations ---------------------------------------------------------------------


def test_a_window_spanning_a_page_break_cites_both_pages():
    regions = [
        _region(_words(6, 0), page="hkb_p0001"),
        _region(_words(6, 6), page="hkb_p0002"),
    ]
    out = ch.split(regions, CFG)
    assert out[0].page_ids == ["hkb_p0001", "hkb_p0002"]


def test_page_ids_are_deduped_and_in_reading_order():
    regions = [_region(_words(2, i * 2), page=f"hkb_p{i:04d}", n=i) for i in range(1, 4)]
    (out,) = ch.split(regions, CFG)
    assert out.page_ids == ["hkb_p0001", "hkb_p0002", "hkb_p0003"]


# --- text fidelity -----------------------------------------------------------------


def test_text_keeps_its_original_case_and_punctuation():
    """MiniLM is uncased: decoding pieces back would lowercase every citation we quote."""
    (out,) = ch.split([_region("Curaçao Jelly — Home-Keeping, p. 214.")], CFG)
    assert out.text == "Curaçao Jelly — Home-Keeping, p. 214."


def test_whitespace_is_normalised_not_preserved():
    (out,) = ch.split([_region("boil  the\nlinen")], CFG)
    assert out.text == "boil the linen"


# --- config ------------------------------------------------------------------------


def test_overlap_is_clamped_below_the_budget():
    """overlap >= budget would repeat a whole window and never reach the end of the page."""
    out = ch.split([_region(_words(40))], {"index": {"chunk_tokens": 12, "overlap": 99}})
    assert out[-1].text.split()[-1] == "w039"


def test_unknown_index_keys_do_not_shadow_defaults():
    merged = ch._merge(ch._DEFAULTS, {"type": "faiss:hnsw", "overlap": 8})
    assert "type" not in merged and merged["overlap"] == 8
    assert merged["chunk_tokens"] == ch._DEFAULTS["chunk_tokens"]


# --- the real tokenizer ------------------------------------------------------------


def test_the_budget_is_counted_in_wordpieces_not_words(monkeypatch):
    """1.32 pieces/word on this corpus, so a word budget would overflow the encoder."""
    monkeypatch.undo()
    tok = pytest.importorskip("transformers").AutoTokenizer
    name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        enc = tok.from_pretrained(name)
    except Exception as exc:  # offline CI has no hub access
        pytest.skip(f"tokenizer unavailable: {exc}")

    text = " ".join(["Home-Keeping curaçao sauceboat"] * 40)
    for c in ch.split([_region(text)], {"index": {"chunk_tokens": 64, "overlap": 8}}):
        assert len(enc(c.text)["input_ids"]) <= 64
