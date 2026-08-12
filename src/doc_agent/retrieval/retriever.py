"""Stage 5 — dense retrieval.

``Retriever`` wraps the ``VectorStore`` built in Stage 4c and the cross-encoder
reranker (``retrieval/rerank.py``) so the agent loop sees a single ``retrieve(query,
k)`` interface.  Everything else — evidence-strength helpers used by ``agent.decide()``
for evidence-gated re-search — lives as module-level helpers so they can be imported
without constructing a Retriever.

Design notes
------------
* The query is embedded via ``embed.encode_texts``, the **same** call path used for
  chunks in Stage 4b, so query and passage vectors always come from identical pooling
  and L2-normalisation.  Using a separate code path for queries is the classic source
  of a silent score-scale mismatch.
* ``Retriever`` is lazy-loaded: the VectorStore is read from disk on the first
  ``retrieve()`` call, not at construction time, so importing the module in tests does
  not require the index to exist on disk.
* ``k`` defaults to ``cfg['retrieve']['k']`` but callers (e.g. the agent re-search
  loop) may pass a larger value to widen the net.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from ..contracts import Chunk
from ..logging_conf import get_logger

if TYPE_CHECKING:
    from ..index.store import VectorStore

log = get_logger(__name__)


class Retriever:
    """Dense retriever backed by the Stage 4 VectorStore.

    Parameters
    ----------
    cfg:
        The full project config dict (``config.load()``).  The ``retrieve``,
        ``index``, ``embed`` and ``device`` keys are read here.
    """

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._rc = cfg["retrieve"]
        self._store: Optional["VectorStore"] = None  # lazy — loaded on first retrieve() call

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_store(self) -> None:
        if self._store is not None:
            return
        from ..index import store as store_mod

        self._store = store_mod.load(self._cfg)

    def _embed_query(self, query: str) -> np.ndarray:
        from ..index.embed import encode_texts

        return encode_texts([query], self._cfg)[0]  # shape (dim,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return the top-``k`` chunks for *query*, scored by cosine similarity.

        Each returned :class:`~doc_agent.contracts.Chunk` has its ``score`` field
        set to the cosine similarity with the query (in ``[-1, 1]``).  This lets
        ``agent.decide()`` call :func:`is_weak` to decide whether evidence-gated
        re-search should widen ``k``.

        Parameters
        ----------
        query:
            The natural-language query string.
        k:
            Number of candidates to return.  Defaults to ``cfg['retrieve']['k']``.
            The agent re-search loop passes ``next_k(k, cfg)`` to widen the net.

        Returns
        -------
        list[Chunk]
            Chunks in descending score order, length ``<= k``.
        """
        if k is None:
            k = int(self._rc["k"])

        self._ensure_store()
        assert self._store is not None, "VectorStore failed to load — run scripts/build_index.sh first"

        # Embed the query in the same space as the passage vectors.
        q_vec = self._embed_query(query).reshape(1, -1).astype(np.float32)

        # Search the vector store.
        results: list[Chunk] = self._store.search(q_vec, k)[0]

        log.info(
            "retrieve: query=%r k=%d -> %d candidates (top score %.3f)",
            query[:60],
            k,
            len(results),
            max((c.score for c in results), default=0.0),
        )

        # Optionally rerank with a cross-encoder.
        # Guard: rerank.rerank() is a stub (raises NotImplementedError) until A3;
        # fall back to dense results so retrieval works end-to-end in A2.
        if self._rc.get("rerank") and results:
            try:
                from .rerank import rerank as _rerank

                results = _rerank(query, results, self._cfg)
                log.info("rerank: %d candidates after reranking", len(results))
            except NotImplementedError:
                log.warning(
                    "rerank stub not yet implemented; returning dense results (%d chunks)",
                    len(results),
                )

        return results


# ---------------------------------------------------------------------------
# Evidence-strength policy helpers — read by agent.decide() for evidence-gated
# re-search (spec: widen k while weak; abstain once k would exceed k_max)
# ---------------------------------------------------------------------------


def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)

def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below ``cfg['retrieve']['weak_threshold']``.

    A cosine score of 0.35 sits between a clear hit (0.65 on the ink-stain
    passage vs its matching query) and a near-miss (0.26) on this corpus.
    The threshold is set in ``configs/config.yaml`` and documented in
    ``index/embed.py``.
    """
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]

def next_k(k: int, cfg: dict) -> int | None:
    """Widen the retrieval net by ``k_step``; return ``None`` to signal ABSTAIN.

    The agent re-search loop calls this when :func:`is_weak` returns ``True``.
    Once ``k + k_step`` would exceed ``k_max`` the function returns ``None``
    and the loop must abstain rather than continue widening.
    """
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None