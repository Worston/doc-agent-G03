"""Stage 4b — embedding: chunk text to unit vectors.

``all-MiniLM-L6-v2`` is a bi-encoder: query and passage go through the *same* tower, so
whatever is done to a chunk here must be done to a query at retrieval time. That is why
``encode_texts`` exists and ``encode`` is a thin wrapper over it — Stage 5 embeds queries
by calling the same function, rather than re-implementing the pooling and the
normalisation slightly differently and quietly skewing the two sides apart.

Vectors are **L2-normalised**, which is the load-bearing choice in this module. It makes
an inner product equal to cosine similarity, so a FAISS inner-product index returns scores
in [-1, 1]. ``retrieve.weak_threshold: 0.35`` is only meaningful against a bounded,
scale-free score: on raw L2 distance the same constant would mean different things for
different chunk lengths, and the agent's evidence gate would fire more or less often
depending on how long a passage happened to be. Spot-checked against three real chunks,
"how do I get ink out of a tablecloth?" scores 0.653 on the ink-stain passage, 0.256 on a
different household instruction and -0.004 on an unrelated one — so the configured 0.35
does sit between a hit and a near-miss on this corpus.

Cost is negligible and the model is cached across calls: 300 texts embed in 0.19 s on CPU,
against ~40 s to construct the encoder the first time.

The configured ``embed.dim`` is checked against what the model actually returns rather
than trusted. A dim mismatch does not raise inside FAISS until search time, and by then
the index on disk is already wrong.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..contracts import Chunk
from ..logging_conf import get_logger

log = get_logger(__name__)

_DEFAULTS: dict = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dim": 384,
    "batch": 64,
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    out.update({k: v for k, v in (over or {}).items() if k in base})
    return out


def _device(want: str) -> str:
    """cuda when it exists, else cpu. ``config.yaml`` ships ``device: cuda``."""
    import torch

    if want == "cuda" and not torch.cuda.is_available():
        log.warning("embed: device 'cuda' unavailable, using 'cpu'")
        return "cpu"
    return want


@lru_cache(maxsize=2)
def _model(name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def encode_texts(texts: list[str], cfg: dict) -> np.ndarray:
    """Texts -> (n, dim) float32 unit vectors. The single encoding path, shared with Stage 5."""
    ec = _merge(_DEFAULTS, cfg.get("embed", {}))
    dim = int(ec["dim"])
    if not texts:
        # FAISS needs the width even with no rows, so an empty result is still (0, dim).
        return np.zeros((0, dim), dtype=np.float32)

    model = _model(str(ec["model"]), _device(str(cfg.get("device", "cpu"))))
    vecs = model.encode(
        texts,
        batch_size=int(ec["batch"]),
        convert_to_numpy=True,
        normalize_embeddings=True,  # inner product == cosine, so scores are in [-1, 1]
        show_progress_bar=False,
    ).astype(np.float32)

    if vecs.shape[1] != dim:
        raise ValueError(
            f"embed.dim is {dim} but {ec['model']!r} returns {vecs.shape[1]}; "
            "the index would be built at the wrong width"
        )
    return vecs


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Chunks -> (n, dim) unit vectors, row i belonging to ``chunks[i]``."""
    vecs = encode_texts([c.text for c in chunks], cfg)
    log.info(
        "embed[%s]: %d chunks -> %s",
        _merge(_DEFAULTS, cfg.get("embed", {}))["model"],
        len(chunks),
        vecs.shape,
    )
    return vecs
