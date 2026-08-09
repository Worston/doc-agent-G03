"""Re-derive the 40 held-out page ids from configs/ alone. Prints; copies nothing.

    python3 grading_kit/heldout_pages/select_heldout.py

The README claims this slice is deterministic from `configs/task.yaml` + `configs/config.yaml
-> seed`. This script IS that claim: running it must reproduce the id list in the README table.

Section derivation (v2). Sections come from the book's own running heads: `CLASS n` inside the
recipe part, the Roman-numeral `Section N` elsewhere, `SUPPLEMENT` for the wartime front
supplement. Heads are forward-filled because verso and recto carry different heads. v1 read the
running head only and matched `[IVX]+` case-sensitively, which mis-assigned every section
OPENER page: an opener carries no running head (its title sits in a decorative masthead), so it
inherited the previous section, and openers that DO carry text print it in caps with OCR-mangled
numerals ("SECTION VIU." for VIII). v2 adds the opener rule below. See README "Correction".
"""

from __future__ import annotations

import random
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
T = yaml.safe_load((REPO / "configs" / "task.yaml").read_text())
EDA, HO, SP = T["eda"], T["heldout"], T["splits"]
SEED = yaml.safe_load((REPO / "configs" / "config.yaml").read_text())["seed"]
EXCL = set(HO["exclude_pages"])
TEST_SECTIONS = set(SP["test_sections"])
VAL_SECTIONS = set(SP["val_sections"])
EYE_OPENERS = {int(k): v for k, v in HO.get("eye_verified_sections", {}).items()}

_P = re.compile(r'<page width="([\d.]+)" height="[\d.]+">(.*?)</page>', re.S)
_W = re.compile(r'<word xMin="([\d.]+)" yMin="[\d.]+" xMax="([\d.]+)"[^>]*>(.*?)</word>')
_CL = re.compile(r"\bCLASS\s+([\d\s]{1,6}?)\s*[—\-–]")
_SU = re.compile(r"\bSUPPLEMENT\b", re.I)
# Running head, e.g. "406  THE HOME-KEEPING BOOK— Section VII— FAMILY DOCTOR".
_SE_HEAD = re.compile(r"\bSec(?:tion|\.)?\s*([IVXUl1|]{1,6})\b", re.I)
# Opener masthead, e.g. "SECTION VIU." — anchored at end of line so that in-body
# cross-references ("... see SECTION V — CONVENIENCES ...", p486) do NOT match.
_SE_OPEN = re.compile(r"\bSECTION\s+([IVXUl1|]{1,6})\s*\.?\s*$", re.I)

ROMAN = {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}
# Glyph confusions this scan is 300 DPI OCR: U<-II, L/1/| <- I.
_FIX = str.maketrans({"L": "I", "1": "I", "|": "I"})


def _pdftotext(*flags: str) -> str:
    return subprocess.run(
        ["pdftotext", *flags, str(REPO / EDA["source_pdf"]), "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _roman(s: str) -> str | None:
    s = s.upper().replace("U", "II").translate(_FIX)
    return s if s in ROMAN else None


def page_stats(xml: str) -> dict[int, dict]:
    st = {}
    for i, (w, body) in enumerate(_P.findall(xml), 1):
        mid = float(w) / 2
        words = _W.findall(body)
        n = len(words)
        cross = sum(1 for a, z, _ in words if float(a) < mid < float(z))
        cf = (cross / n) if n else 1.0
        st[i] = {
            "words": n,
            "cross": cf,
            "layout": ("two_column" if cf < EDA["two_column_max_cross_frac"] else "single_column"),
        }
    return st


def section_map(pages: list[str], st: dict[int, dict]) -> dict[int, str | None]:
    def lines(p: int) -> list[str]:
        return [ln.strip() for ln in pages[p - 1].splitlines() if ln.strip()]

    def head(p: int) -> str:
        return next((s for s in lines(p) if len(s) >= 6), "")

    def klass(p: int) -> str | None:
        m = _CL.search(head(p))
        if not m:
            return None
        d = re.sub(r"\s+", "", m.group(1))
        return f"CLASS {int(d):02d}" if d.isdigit() and 1 <= int(d) <= 40 else None

    def sec(p: int) -> str | None:
        if p in EYE_OPENERS:
            return EYE_OPENERS[p]
        for ln in lines(p)[:5]:
            m = _SE_OPEN.search(ln)
            if m and (r := _roman(m.group(1))):
                return f"Section {r}"
        h = head(p)
        if _SU.search(h):
            return "SUPPLEMENT"
        m = _SE_HEAD.search(h)
        if m and (r := _roman(m.group(1))):
            return f"Section {r}"
        return None

    n = len(pages)
    classes = [klass(p) for p in range(1, n + 1)]
    sections = [sec(p) for p in range(1, n + 1)]
    lo = min(i for i, c in enumerate(classes) if c)
    hi = max(i for i, c in enumerate(classes) if c)
    unit, fc, fs = {}, None, None
    for i in range(n):
        fc = classes[i] or fc
        fs = sections[i] or fs
        unit[i + 1] = fc if lo <= i <= hi and fc else fs
    return unit


def blocks(seq: list[int], k: int) -> list[list[int]]:
    n = len(seq)
    return [seq[(j * n) // k : ((j + 1) * n) // k] for j in range(k)]


def main() -> int:
    pages = _pdftotext("-layout").split("\f")
    st = page_stats(_pdftotext("-bbox-layout"))
    unit = section_map(pages, st)

    content = [
        p
        for p in sorted(st)
        if st[p]["words"] >= EDA["content_min_words"] and p not in EXCL and unit[p] is not None
    ]
    n_te = sum(1 for p in content if unit[p] in TEST_SECTIONS)
    n_va = sum(1 for p in content if unit[p] in VAL_SECTIONS)
    n_tr = len(content) - n_te - n_va
    print(
        f"content pages {len(content)}   train {n_tr} ({n_tr/len(content):.1%})"
        f"  val {n_va} ({n_va/len(content):.1%})  test {n_te} ({n_te/len(content):.1%})"
    )

    # Layout quota mirrors the CORPUS share, not the test split's own share, so the pooled
    # character-F1 generalises to the corpus rather than to whichever sections landed in test.
    corpus_two = sum(1 for p in content if st[p]["layout"] == "two_column") / len(content)
    n_two = round(HO["n_pages"] * corpus_two)
    pool = sorted(p for p in content if unit[p] in TEST_SECTIONS)
    tw = [p for p in pool if st[p]["layout"] == "two_column"]
    sg = [p for p in pool if st[p]["layout"] == "single_column"]
    print(
        f"corpus two-column {corpus_two:.1%} -> quota two={n_two} single={HO['n_pages']-n_two}"
        f"   test pool {len(pool)} ({len(tw)} two-col)"
    )

    # Cut each stratum into n contiguous blocks and draw one per block, so the slice spreads
    # over all test sections instead of clumping in the largest.
    rng = random.Random(SEED)
    picked: list[int] = []
    for stratum, k in ((tw, n_two), (sg, HO["n_pages"] - n_two)):
        for blk in blocks(stratum, k):
            cands = [p for p in blk if all(abs(p - q) > 1 for q in picked)] or blk
            picked.append(rng.choice(cands))
    picked.sort()

    print("\n| page_id | pdf page | section | words | cross% |")
    print("|---|---|---|---|---|")
    for p in picked:
        print(
            f"| hkb_p{p:04d} | {p} | {unit[p]} | {st[p]['words']} " f"| {st[p]['cross']*100:.2f} |"
        )

    ws = sorted(st[p]["words"] for p in picked)
    two = sum(1 for p in picked if st[p]["layout"] == "two_column")
    print(
        f"\nselected {len(picked)} unique={len(set(picked))}  two_column={two} "
        f"({two/len(picked):.1%})   words {ws[0]}-{ws[-1]} median {ws[len(ws)//2]}"
    )
    print(f"sections covered: {sorted({unit[p] for p in picked})}")
    off = [p for p in picked if unit[p] not in TEST_SECTIONS]
    print(f"pages outside a TEST section: {off or 'none'}")
    print("\nSELECTED =", picked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
