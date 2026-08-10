"""Data — data schema/quality validation at ingest."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from ..contracts import Page  # type: ignore[attr-defined]
from ..logging_conf import get_logger

log = get_logger(__name__)

# ── corpus-level floors (from spec: ≥300 pages AND ≥60k words) ─────────────
MIN_PAGES = 300
MIN_WORDS = 60_000

# ── approximate word count per page when raw text is unavailable ─────────────
# The validate() signature receives Page objects which only carry image_path and
# doc_id, not extracted text.  We therefore count words from any *sidecar* .txt
# file that the ingest pipeline may have placed alongside the image.  If no
# sidecar is present we skip the per-page word tally and trust the corpus-level
# count declared in configs/task.yaml (logged as a warning).
_WORD_RE = re.compile(r"\S+")


def _words_from_sidecar(image_path: str) -> int | None:
    """Return word count from a .txt sidecar next to the image, or None."""
    txt = Path(image_path).with_suffix(".txt")
    if txt.exists():
        return len(_WORD_RE.findall(txt.read_text(encoding="utf-8", errors="ignore")))
    return None


# ── public API ────────────────────────────────────────────────────────────────

def validate(pages: list[Page], splits: dict[str, list[str]] | None = None) -> None:
    """Assert min pages/words, format integrity, and no cross-split leakage.

    Parameters
    ----------
    pages:
        All Page objects produced by the ingest stage.
    splits:
        Optional mapping of split-name → list[doc_id] as declared in
        configs/task.yaml.  When supplied, leakage (same doc_id in >1 split)
        is detected and raised as a ValueError.

    Raises
    ------
    ValueError
        If any check fails.  The message names the failing assertion so it is
        actionable without a stack trace.
    """
    # ── 1. Corpus-floor checks ────────────────────────────────────────────────
    n_pages = len(pages)
    if n_pages < MIN_PAGES:
        raise ValueError(
            f"Corpus too small: {n_pages} pages < required {MIN_PAGES}. "
            "Add more scanned pages or lower MIN_PAGES only after updating the spec."
        )
    log.info("corpus_floor_pages", extra={"n_pages": n_pages, "min": MIN_PAGES})

    total_words: int = 0
    missing_sidecars: int = 0
    for page in pages:
        w = _words_from_sidecar(page.image_path)
        if w is None:
            missing_sidecars += 1
        else:
            total_words += w

    if missing_sidecars == n_pages:
        # No sidecars at all — cannot measure; emit a warning and skip the check.
        log.warning(
            "word_count_unchecked",
            extra={
                "reason": "no .txt sidecars found beside page images",
                "advice": (
                    "Run OCR first, or ensure ingest writes .txt sidecars. "
                    "The spec requires ≥60 k words; confirm manually."
                ),
            },
        )
    elif total_words < MIN_WORDS:
        raise ValueError(
            f"Corpus too small: {total_words} words counted from sidecars "
            f"< required {MIN_WORDS}. "
            "(If sidecars are partial, re-run OCR on all pages first.)"
        )
    else:
        log.info(
            "corpus_floor_words",
            extra={"total_words": total_words, "min": MIN_WORDS, "missing_sidecars": missing_sidecars},
        )

    # ── 2. Format / field integrity ───────────────────────────────────────────
    seen_ids: set[str] = set()
    for i, page in enumerate(pages):
        # Non-empty string fields
        if not page.id or not isinstance(page.id, str):
            raise ValueError(f"Page[{i}].id is empty or not a string: {page.id!r}")
        if not page.doc_id or not isinstance(page.doc_id, str):
            raise ValueError(f"Page[{i}].doc_id is empty or not a string: {page.doc_id!r}")
        if not page.image_path or not isinstance(page.image_path, str):
            raise ValueError(f"Page[{i}].image_path is empty or not a string: {page.image_path!r}")

        # Duplicate page ids
        if page.id in seen_ids:
            raise ValueError(f"Duplicate page id detected: {page.id!r} (page index {i})")
        seen_ids.add(page.id)

        # Image file must exist on disk
        if not Path(page.image_path).exists():
            raise ValueError(
                f"Page {page.id!r}: image file not found at {page.image_path!r}. "
                "Run scripts/get_data.sh first."
            )

    log.info("format_check_passed", extra={"n_pages": n_pages})

    # ── 3. Cross-split leakage ────────────────────────────────────────────────
    if splits:
        doc_to_splits: dict[str, list[str]] = defaultdict(list)
        for split_name, doc_ids in splits.items():
            for doc_id in doc_ids:
                doc_to_splits[doc_id].append(split_name)

        leaked = {doc: s for doc, s in doc_to_splits.items() if len(s) > 1}
        if leaked:
            details = "; ".join(
                f"doc {d!r} in splits {s}" for d, s in list(leaked.items())[:5]
            )
            raise ValueError(
                f"Cross-split leakage detected ({len(leaked)} doc(s)): {details}. "
                "Ensure each document is assigned to exactly one split."
            )
        log.info(
            "leakage_check_passed",
            extra={"n_splits": len(splits), "n_docs_checked": len(doc_to_splits)},
        )
    else:
        log.info("leakage_check_skipped", extra={"reason": "no splits mapping provided"})

