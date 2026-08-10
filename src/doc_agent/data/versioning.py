"""Data — corpus versioning (which corpus version → which result)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ..logging_conf import get_logger

log = get_logger(__name__)

# Where the version log is written.  One JSON-Lines record per snapshot call.
_VERSION_LOG = Path("data/corpus_versions.jsonl")

# Files with these extensions are hashed.  Add more as needed.
_HASH_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf", ".txt"}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def snapshot(corpus_dir: str) -> str:
    """Hash every file in *corpus_dir* and record a deterministic version id.

    The function walks *corpus_dir* recursively, hashes all files whose
    extension is in ``_HASH_EXTS`` (sorted by relative path for
    reproducibility), combines the per-file digests into a single corpus-level
    SHA-256, and writes a JSON-Lines record to ``data/corpus_versions.jsonl``.

    Parameters
    ----------
    corpus_dir:
        Path to the directory that holds the scanned page-images (e.g.
        ``data/raw``).

    Returns
    -------
    str
        A 64-character hex SHA-256 that uniquely identifies this corpus state.
        The same files in the same directory always produce the same id, so the
        id is deterministic and auditable.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise FileNotFoundError(
            f"corpus_dir not found: {root.resolve()!s}. "
            "Run scripts/get_data.sh first."
        )

    # Collect and sort paths deterministically.
    paths = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _HASH_EXTS
    )
    if not paths:
        raise ValueError(
            f"No hashable files (extensions {_HASH_EXTS}) found in {root}. "
            "Corpus directory may be empty."
        )

    log.info("corpus_snapshot_start", extra={"corpus_dir": str(root), "n_files": len(paths)})

    # Hash each file and feed its (relative-path, digest) pair into a combiner.
    combiner = hashlib.sha256()
    file_digests: list[dict] = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        digest = _sha256_file(p)
        combiner.update(f"{rel}:{digest}\n".encode())
        file_digests.append({"rel_path": rel, "sha256": digest})

    corpus_id = combiner.hexdigest()

    # Persist the record.
    record = {
        "version_id": corpus_id,
        "corpus_dir": str(root.resolve()),
        "n_files": len(paths),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": file_digests,
    }

    _VERSION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _VERSION_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    log.info(
        "corpus_snapshot_done",
        extra={"version_id": corpus_id, "log": str(_VERSION_LOG)},
    )
    return corpus_id


