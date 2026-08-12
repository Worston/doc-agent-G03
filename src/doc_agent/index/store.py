"""Stage 4c — the vector store: persist chunks + vectors, hand Stage 5 a searchable object.

Search is an **inner product**, not an L2 distance, because Stage 4b already normalises every
vector: on unit vectors an inner product *is* cosine, so the scores Stage 5 compares against
``retrieve.weak_threshold`` arrive bounded in [-1, 1] with no rescaling anywhere.

Three files are written, not one:

* ``vectors.npy``  — the matrix, row-aligned with
* ``chunks.jsonl`` — the text, ids and ``page_ids``, because a search hit is an integer and an
  integer cannot be cited;
* ``meta.json``    — the embedding model and width the vectors were produced with.

The third earns its keep: change ``embed.model`` without rebuilding and a stale index still
loads, still searches and still returns confidently wrong neighbours, because nothing about a
vector says which encoder made it. ``load`` refuses an index whose meta disagrees with config.

Backends, chosen by ``index.type``:

* ``numpy:flat`` (default) — one matmul, exact.
* ``faiss:flat`` / ``faiss:hnsw`` — built at load time from the same ``vectors.npy``, so the
  on-disk format does not depend on which backend is configured.

The default is numpy for two measured reasons. **faiss cannot share this process with torch**:
torch, sklearn and faiss each ship their own ``libomp.dylib``, and the second one to
initialise aborts with ``OMP: Error #15`` and takes the interpreter down with SIGSEGV — which
would kill ``build_knowledge_base()``, since it embeds and then indexes in one process. The
only workaround that holds is ``OMP_NUM_THREADS=1``, which also single-threads torch and so
taxes the training path. And faiss buys nothing here to pay for that: over 8 queries against
this corpus, numpy exact runs 0.004 ms/query against faiss:flat 0.009 and faiss:hnsw 0.012, at
recall@10 = 1.000 for all three. numpy holds 0.125 ms/query at 2k vectors and 0.25 ms at 20k,
roughly ten times this book. HNSW is an approximation that pays off in the millions; this book
is ~1.8k windows, where the exact scan is already the fast option. Nothing is removed — set
``index.type`` to a ``faiss:*`` value (with ``OMP_NUM_THREADS=1``) and it is used instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import Chunk
from ..logging_conf import get_logger

log = get_logger(__name__)

_DEFAULTS: dict = {
    "type": "numpy:flat",
    "out_dir": "data/index",
    "hnsw_m": 32,  # faiss:hnsw only — neighbours per node
    "ef_construction": 200,  # faiss:hnsw only — build-time candidate list
    "ef_search": 64,  # faiss:hnsw only — query-time candidates; raise to trade latency for recall
}

_KINDS = ("numpy:flat", "faiss:flat", "faiss:hnsw")


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in (over or {}).items() if k in base})
    return out


def _paths(cfg: dict) -> tuple[Path, Path, Path, Path]:
    out = Path(_merge(_DEFAULTS, cfg.get("index", {}))["out_dir"])
    return out, out / "vectors.npy", out / "chunks.jsonl", out / "meta.json"


def _faiss_index(kind: str, vecs: np.ndarray, ic: dict) -> Any:
    """A faiss index over ``vecs``. Imported here, never at module scope — see the docstring."""
    import faiss

    index: Any
    if kind == "faiss:flat":
        index = faiss.IndexFlatIP(vecs.shape[1])
    else:
        index = faiss.IndexHNSWFlat(vecs.shape[1], int(ic["hnsw_m"]), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(ic["ef_construction"])
        index.hnsw.efSearch = int(ic["ef_search"])
    index.add(vecs)
    return index


@dataclass
class VectorStore:
    """Vectors plus the chunks their rows stand for. ``index`` is None for the numpy backend."""

    vectors: np.ndarray
    chunks: list[Chunk]
    index: Any = None

    def search(self, queries: np.ndarray, k: int) -> list[list[Chunk]]:
        """Query vectors -> per-query chunks, best first, each carrying its cosine score."""
        queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        if not self.chunks:
            return [[] for _ in queries]
        k = min(k, len(self.chunks))

        if self.index is not None:
            scores, ids = self.index.search(queries, k)
        else:
            sims = queries @ self.vectors.T
            ids = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sims, ids, 1), axis=1)
            ids = np.take_along_axis(ids, order, 1)
            scores = np.take_along_axis(sims, ids, 1)

        # Copies, not the stored objects: a score belongs to one query, and mutating the shared
        # chunks would leave the previous query's numbers on them.
        return [
            [
                self.chunks[i].model_copy(update={"score": float(s)})
                for i, s in zip(irow, srow, strict=True)
                if i >= 0
            ]
            for irow, srow in zip(ids, scores, strict=True)
        ]


def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Write vectors, the chunk sidecar and the provenance meta to ``index.out_dir``."""
    ic = _merge(_DEFAULTS, cfg.get("index", {}))
    if str(ic["type"]) not in _KINDS:
        raise ValueError(f"unknown index.type {ic['type']!r} (expected one of {_KINDS})")
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors; rows would misalign")
    if not chunks:
        raise ValueError("refusing to build an empty index: no chunks reached Stage 4")

    vecs = np.ascontiguousarray(vectors, dtype=np.float32)
    out, vec_p, chunk_p, meta_p = _paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    np.save(vec_p, vecs)
    chunk_p.write_text("\n".join(c.model_dump_json() for c in chunks) + "\n")
    meta_p.write_text(
        json.dumps(
            {
                "type": ic["type"],
                "dim": int(vecs.shape[1]),
                "count": len(chunks),
                "embed_model": cfg.get("embed", {}).get("model", ""),
            },
            indent=2,
        )
    )
    log.info("index: %d vectors (dim %d) -> %s", len(chunks), vecs.shape[1], out)


def load(cfg: dict) -> VectorStore:
    """Read the index back, refusing one that was built with a different embedder."""
    ic = _merge(_DEFAULTS, cfg.get("index", {}))
    out, vec_p, chunk_p, meta_p = _paths(cfg)
    if not vec_p.exists():
        raise FileNotFoundError(f"no index at {out}; run scripts/build_index.sh first")

    want_model = cfg.get("embed", {}).get("model", "")
    meta = json.loads(meta_p.read_text())
    if meta.get("embed_model") != want_model:
        raise ValueError(
            f"index at {out} was built with {meta.get('embed_model')!r} but embed.model is "
            f"{want_model!r}; those vectors are not comparable to this model's queries — rebuild"
        )

    vecs = np.ascontiguousarray(np.load(vec_p), dtype=np.float32)
    lines = [ln for ln in chunk_p.read_text().splitlines() if ln.strip()]
    chunks = [Chunk.model_validate_json(ln) for ln in lines]
    if len(chunks) != len(vecs):
        raise ValueError(f"{len(chunks)} chunks but {len(vecs)} vectors at {out}; rebuild")

    kind = str(ic["type"])
    index = _faiss_index(kind, vecs, ic) if kind.startswith("faiss:") else None
    log.info("index: loaded %d vectors (%s) from %s", len(chunks), kind, out)
    return VectorStore(vectors=vecs, chunks=chunks, index=index)
