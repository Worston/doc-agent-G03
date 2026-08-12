"""Stage 4 tests — CI runs these.

Chunking is where a citation is either preserved or lost, so what is tested is the
bookkeeping this module owns: that windows respect the wordpiece budget, that they
overlap and always advance, that a window straddling a page break remembers both pages,
and that the text is never round-tripped through the tokenizer.
"""

import numpy as np
import pytest

from doc_agent.contracts import Chunk
from doc_agent.index import chunk as ch
from doc_agent.index import embed as em

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


# ===================================================================================
# embedding
# ===================================================================================

ECFG = {"embed": {"model": "stub", "dim": 4}}


def _fake_encoder(monkeypatch, width: int = 4):
    """An encoder that returns a recognisable row per text, so alignment is checkable."""

    class _Stub:
        def encode(self, texts, **kw):
            return np.array([[float(len(t))] + [0.0] * (width - 1) for t in texts])

    monkeypatch.setattr(em, "_model", lambda name, device: _Stub())


def test_encoding_returns_one_row_per_chunk_in_order(monkeypatch):
    _fake_encoder(monkeypatch)
    chunks = [
        Chunk(id=f"hkb#c{i}", doc_id="hkb", text="x" * n, page_ids=["hkb_p0001"])
        for i, n in enumerate((3, 1, 2))
    ]
    vecs = em.encode(chunks, ECFG)
    assert vecs.shape == (3, 4)
    assert list(vecs[:, 0]) == [3.0, 1.0, 2.0], "row i must belong to chunk i"


def test_vectors_are_float32_for_faiss(monkeypatch):
    """faiss silently copies anything else; float64 doubles the index for no gain."""
    _fake_encoder(monkeypatch)
    assert em.encode_texts(["a"], ECFG).dtype == np.float32


def test_no_chunks_still_yields_the_index_width():
    """faiss needs the dimension even with nothing to add, so (0, dim), not (0,)."""
    assert em.encode_texts([], ECFG).shape == (0, 4)


def test_a_dim_mismatch_fails_loudly(monkeypatch):
    """Otherwise the wrong-width index is written and only fails at search time."""
    _fake_encoder(monkeypatch, width=8)
    with pytest.raises(ValueError, match="embed.dim is 4"):
        em.encode_texts(["a"], ECFG)


def test_unknown_embed_keys_do_not_shadow_defaults():
    merged = em._merge(em._DEFAULTS, {"dim": 768, "type": "faiss:hnsw"})
    assert "type" not in merged and merged["dim"] == 768
    assert merged["batch"] == em._DEFAULTS["batch"]


def test_cuda_is_downgraded_when_the_machine_has_none():
    """config.yaml ships device: cuda; this Mac has none and must not crash on it."""
    torch = pytest.importorskip("torch")
    want = "cuda" if torch.cuda.is_available() else "cpu"
    assert em._device("cuda") == want


# --- the real encoder --------------------------------------------------------------


@pytest.fixture(scope="module")
def real_cfg():
    cfg = {"embed": {"model": "sentence-transformers/all-MiniLM-L6-v2", "dim": 384}}
    try:
        em.encode_texts(["warm the cache"], cfg)
    except Exception as exc:  # offline CI has no hub access
        pytest.skip(f"embedding model unavailable: {exc}")
    return cfg


def test_rows_are_unit_vectors_so_inner_product_is_cosine(real_cfg):
    """retrieve.weak_threshold is a constant in [-1, 1]; unnormalised scores break it."""
    vecs = em.encode_texts(["boil the linen", "a quite different sentence"], real_cfg)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_a_query_is_closer_to_its_answer_than_to_an_unrelated_chunk(real_cfg):
    passages = [
        "To remove ink stains from linen, wet the spot with lemon juice and salt.",
        "The dining-room lights were all in operation, each consuming five amperes.",
    ]
    vecs = em.encode_texts(passages, real_cfg)
    query = em.encode_texts(["how do I get ink out of a tablecloth?"], real_cfg)
    ink, lamps = (query @ vecs.T)[0]
    assert ink > real_cfg.get("retrieve", {}).get("weak_threshold", 0.35) > lamps


def test_queries_and_passages_share_one_encoder(real_cfg):
    """A bi-encoder is only comparable if both sides took the identical path."""
    text = "Take a pound of loaf sugar and boil it until it ropes."
    as_chunk = em.encode([Chunk(id="c", doc_id="hkb", text=text, page_ids=["p"])], real_cfg)
    assert np.array_equal(as_chunk[0], em.encode_texts([text], real_cfg)[0])
