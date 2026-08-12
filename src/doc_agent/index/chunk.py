"""Stage 4a — chunking: Stage 3's regions become fixed-size retrieval windows.

Stage 3 emits one chunk per layout region, which is the right unit for *ordering* but the
wrong unit for *retrieval*. Measured over 8 held-out pages: 96 regions, median 47 tokens,
39 of them under 32 — a recipe title, its yield line and its method are three separate
regions, so embedding regions directly would scatter one recipe across three vectors and
leave most of them too short to carry meaning. So the regions of a document are
concatenated back into one stream, in the reading order Stage 2 established, and then cut
into overlapping windows.

Windows are budgeted in **wordpieces, not words**. The same pages tokenise at 1.32
MiniLM pieces per word, so a 256-*word* window is ~338 pieces and the encoder would
silently truncate the tail of every chunk. The budget is therefore measured with the
embedding model's own tokenizer.

The cut itself is made on **word boundaries in the original text**, never by decoding
wordpieces back to a string: ``all-MiniLM-L6-v2`` is uncased, so a decode round-trip would
lowercase the text and mangle it, and this text is what the answer quotes and what a
reranker sees. Tokenisation is used to count, not to rewrite.

``page_ids`` accumulates every page a window touches, in order, so a chunk that straddles
a page break can still be cited to both.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..contracts import Chunk
from ..logging_conf import get_logger

log = get_logger(__name__)

_DEFAULTS: dict = {
    "chunk_tokens": 256,  # window size in wordpieces, incl. the [CLS]/[SEP] MiniLM adds
    "overlap": 32,  # trailing wordpieces repeated in the next window
}

_SPECIALS = 2  # [CLS] ... [SEP]


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in (over or {}).items() if k in base})
    return out


@lru_cache(maxsize=2)
def _tokenizer(name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def _word_lengths(words: list[str], model: str) -> list[int]:
    """Wordpieces each word costs. One batched call, not one call per word."""
    if not words:
        return []
    encoded = _tokenizer(model)(words, add_special_tokens=False)["input_ids"]
    # A word can tokenise to nothing (a stray combining mark); it still occupies the text,
    # so charge it 1 rather than letting a window grow without bound.
    return [max(1, len(ids)) for ids in encoded]


def _windows(lengths: list[int], budget: int, overlap: int) -> list[tuple[int, int]]:
    """Half-open word index ranges, each <= ``budget`` pieces, consecutive ones overlapping."""
    spans: list[tuple[int, int]] = []
    start, n = 0, len(lengths)
    while start < n:
        end, used = start, 0
        # ``end == start`` lets a single over-budget word through rather than looping forever.
        while end < n and (used + lengths[end] <= budget or end == start):
            used += lengths[end]
            end += 1
        spans.append((start, end))
        if end >= n:
            break
        back, taken = end, 0
        while back > start + 1 and taken + lengths[back - 1] <= overlap:
            taken += lengths[back - 1]
            back -= 1
        start = back  # > start, so the loop always advances
    return spans


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Region chunks -> overlapping token windows, one document at a time."""
    ic = _merge(_DEFAULTS, cfg.get("index", {}))
    model = str(cfg.get("embed", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2"))
    budget = max(1, int(ic["chunk_tokens"]) - _SPECIALS)
    overlap = max(0, min(int(ic["overlap"]), budget - 1))

    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    out: list[Chunk] = []
    for doc_id, doc_chunks in by_doc.items():
        words: list[str] = []
        pages: list[str] = []
        for c in doc_chunks:
            page = c.page_ids[0] if c.page_ids else doc_id
            for w in c.text.split():
                words.append(w)
                pages.append(page)

        lengths = _word_lengths(words, model)
        for n, (a, b) in enumerate(_windows(lengths, budget, overlap)):
            out.append(
                Chunk(
                    id=f"{doc_id}#c{n:05d}",
                    doc_id=doc_id,
                    text=" ".join(words[a:b]),
                    page_ids=list(dict.fromkeys(pages[a:b])),
                )
            )

    log.info(
        "chunk: %d regions -> %d windows of <=%d pieces (overlap %d) over %d doc(s)",
        len(chunks),
        len(out),
        budget,
        overlap,
        len(by_doc),
    )
    return out
