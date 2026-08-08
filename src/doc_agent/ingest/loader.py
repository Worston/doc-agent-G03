"""Stage 1 — load scanned page-images"""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Page
from ..logging_conf import get_logger

log = get_logger(__name__)

# pdftoppm writes "<prefix>-350.png"; match the LAST run of digits in the stem.
_PAGE_NUM = re.compile(r"(\d+)(?!.*\d)")


def _page_number(path: Path) -> int:
    m = _PAGE_NUM.search(path.stem)
    if m is None:
        raise ValueError(f"no page number in filename: {path}")
    return int(m.group(1))


def load_pages(cfg: dict) -> list[Page]:
    """Read data/raw/ -> list[Page], ordered by page number.

    Page ids are ``<doc_id>_p<NNNN>`` (e.g. ``hkb_p0350``) — the citation unit
    declared in the A1 data schema, so every chunk traces back to one scan.
    """
    ic = cfg["ingest"]
    raw_dir = Path(ic["raw_dir"])
    doc_id = str(ic["doc_id"])

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"{raw_dir} not found — run scripts/get_data.sh first")

    files = sorted(raw_dir.glob(ic["page_glob"]), key=_page_number)
    if not files:
        raise FileNotFoundError(
            f"no images matching {ic['page_glob']!r} in {raw_dir} — run scripts/get_data.sh first"
        )

    pages = [
        Page(id=f"{doc_id}_p{_page_number(f):04d}", image_path=str(f), doc_id=doc_id) for f in files
    ]

    if len({p.id for p in pages}) != len(pages):
        raise ValueError(f"duplicate page numbers in {raw_dir} — clear it and re-run get_data.sh")

    log.info("loaded %d page images from %s", len(pages), raw_dir)
    return pages
