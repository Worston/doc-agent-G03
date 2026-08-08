# Held-out pages — the OCR oracle

Page-IMAGES set aside and **never** OCR-trained or OCR-tuned on. A grader authors fresh
questions from these pages and checks answers against `../labels.jsonl`.

40 pages, as committed in A1. Files are named by `page_id` (`hkb_pNNNN.png`, `NNNN` = PDF page
number, matching the id scheme `ingest/loader.py` assigns) so they join directly to the
`page_id` key in `../labels.jsonl`. Each file is **byte-identical** to its
`data/raw/homekeepingbook00brow-NNN.png` counterpart, so a score measured here is the score the
pipeline would get on the real corpus — no rescaling or recompression in between.

## How these 40 were chosen

Deterministic from `configs/task.yaml → heldout` + `splits`, plus `configs/config.yaml →
seed: 42`. Re-running the rule reproduces exactly this list.

**1. Section-level split first.** A1 committed to *"Split unit = SECTION (the book's own
Class/Part divisions), not page. Approximately 70% train / 15% validation / 15% test by section,
assigned whole."* Sections are recovered from the running heads — `CLASS n` inside the recipe
part (pp. 219–423), the Roman-numeral `Section N` elsewhere, `SUPPLEMENT` for the wartime front
supplement. Two wrinkles had to be handled: verso and recto carry **different** running heads
(verso repeats the section, recto names the class), so heads are forward-filled; and the class
digits arrive space-split from OCR, e.g. `CLASS 1 6— VEGETABLES` is Class 16 — which is the very
boundary A1 cites as its example. Realised on 491 content pages:

| split | content pages | share | sections |
|---|---|---|---|
| train | 333 | 67.8% | 24 |
| validation | 79 | 16.1% | 6 |
| test | 79 | 16.1% | 8 |

**2. All 40 are drawn from TEST sections only** — `CLASS 01, 07, 12, 14, 19, 20`, `Section II`,
`Section VII`. This is what makes "never trained or tuned on" true at the section level rather
than merely the page level.

**3. Content pages only** — `words >= eda.content_min_words` (80). Sparse front-matter, plates
and blank memo pages cannot serve as an OCR oracle.

**4. Contents and index pages removed** — `heldout.exclude_pages`. These clear the word floor
but are lists of page numbers, not prose, so a grader cannot author a real question from them.
pp. 7 / 93 / 189 are contents pages, pp. 604–614 the back index; A1 already committed to
excluding the alphabetical index, so this follows A1 rather than adding a new rule. Note p112
was flagged by the same scan but is a genuine two-column *budget table* and is deliberately
kept — numeric tables are hard OCR and squarely on-domain.

**5. Layout quota mirrors the corpus, not the split.** 14 two-column / 26 single-column. The
test split is 38% two-column but the corpus is 33.8%, so drawing at the corpus rate keeps the
pooled character-F1 representative of the corpus. Per-layout numbers are reported separately
(A1 Section 4), so nothing is lost.

**6. Spread within each stratum** — each stratum is cut into *n* contiguous blocks and one page
drawn per block, so the slice covers all eight test sections rather than clumping in the
largest. No minimum page gap is enforced: adjacent pages inside the slice are harmless because
both sides are held out, and leakage is prevented by whole-section assignment.

## What the slice looks like

| property | held-out (40) | content corpus (491) |
|---|---|---|
| two-column share | 35.0% (14) | 33.8% (166) |
| recipe-class pages | 42% (17) | 36.0% |
| median words/page | 685 | 641 |
| word range | 168 – 1,038 | 86 – 1,064 |
| median speckles/megapixel | 77.5 | 66.3 |
| speckle p10 / p90 | 19.0 / 121.4 | 10.8 / 135.6 |

Speckles/megapixel counts connected components of <= 4 px after Otsu binarisation — a proxy for
dust, bleed-through and scan dirt. It is reported because *degraded scans* is our secondary data
condition, and a held-out set that happened to be clean would flatter the OCR score. It does not:
the slice sits slightly above the corpus median.

## The 40 pages

`layout` is the centre-crossing heuristic from `notebooks/eda.ipynb`; `cross%` is the share of
word boxes straddling the page centre-line (< 1.2% ⇒ two-column).

| page_id | pdf page | section | words | cross% | layout |
|---|---|---|---|---|---|
| hkb_p0124 | 124 | Section II | 788 | 2.03 | single_column |
| hkb_p0125 | 125 | Section II | 927 | 0.76 | two_column |
| hkb_p0127 | 127 | Section II | 852 | 4.46 | single_column |
| hkb_p0131 | 131 | Section II | 526 | 4.94 | single_column |
| hkb_p0134 | 134 | Section II | 680 | 3.38 | single_column |
| hkb_p0137 | 137 | Section II | 708 | 4.52 | single_column |
| hkb_p0139 | 139 | Section II | 358 | 6.15 | single_column |
| hkb_p0141 | 141 | Section II | 355 | 5.07 | single_column |
| hkb_p0220 | 220 | CLASS 01 | 940 | 0.11 | two_column |
| hkb_p0223 | 223 | CLASS 01 | 504 | 5.56 | single_column |
| hkb_p0253 | 253 | CLASS 07 | 1012 | 0.10 | two_column |
| hkb_p0255 | 255 | CLASS 07 | 410 | 0.49 | two_column |
| hkb_p0257 | 257 | CLASS 07 | 690 | 4.49 | single_column |
| hkb_p0258 | 258 | CLASS 07 | 477 | 0.21 | two_column |
| hkb_p0260 | 260 | CLASS 07 | 625 | 3.68 | single_column |
| hkb_p0276 | 276 | CLASS 12 | 962 | 0.10 | two_column |
| hkb_p0281 | 281 | CLASS 14 | 931 | 4.19 | single_column |
| hkb_p0283 | 283 | CLASS 14 | 546 | 4.58 | single_column |
| hkb_p0285 | 285 | CLASS 14 | 291 | 8.25 | single_column |
| hkb_p0327 | 327 | CLASS 19 | 942 | 4.46 | single_column |
| hkb_p0332 | 332 | CLASS 20 | 939 | 1.28 | single_column |
| hkb_p0334 | 334 | CLASS 20 | 168 | 0.60 | two_column |
| hkb_p0337 | 337 | CLASS 20 | 784 | 1.53 | single_column |
| hkb_p0338 | 338 | CLASS 20 | 999 | 0.10 | two_column |
| hkb_p0341 | 341 | CLASS 20 | 808 | 0.37 | two_column |
| hkb_p0472 | 472 | Section VII | 469 | 5.76 | single_column |
| hkb_p0474 | 474 | Section VII | 834 | 1.08 | two_column |
| hkb_p0477 | 477 | Section VII | 636 | 4.56 | single_column |
| hkb_p0479 | 479 | Section VII | 176 | 4.55 | single_column |
| hkb_p0482 | 482 | Section VII | 1038 | 2.12 | single_column |
| hkb_p0484 | 484 | Section VII | 992 | 4.54 | single_column |
| hkb_p0485 | 485 | Section VII | 558 | 0.54 | two_column |
| hkb_p0488 | 488 | Section VII | 844 | 2.37 | single_column |
| hkb_p0489 | 489 | Section VII | 946 | 4.02 | single_column |
| hkb_p0492 | 492 | Section VII | 674 | 0.15 | two_column |
| hkb_p0495 | 495 | Section VII | 438 | 0.23 | two_column |
| hkb_p0497 | 497 | Section VII | 572 | 4.72 | single_column |
| hkb_p0499 | 499 | Section VII | 458 | 4.80 | single_column |
| hkb_p0503 | 503 | Section VII | 825 | 0.00 | two_column |
| hkb_p0507 | 507 | Section VII | 370 | 4.86 | single_column |

p482 is one of the three pages A1 names as worst-case hyphenation (44 line-break hyphens), so
the slice deliberately contains a known-hard page rather than avoiding it.

## Honest limits of this split

- **Same book, same typeface, same scanning session.** A1 states this openly: the assignment
  asks for a split by DOCUMENT and we have one document, so section is the strictest correlation
  unit available. No single-corpus split can measure generalisation to an unseen press or
  scanner. The number from this slice is *in-domain* OCR accuracy and must be described so.
- **Section boundaries are themselves OCR-derived.** They come from running heads, which are
  damaged on part of the corpus; heads were directly readable on 330 pages and forward-filled
  elsewhere. A misplaced boundary would move a page between splits.
- **`layout` here is the text-layer heuristic, not ground truth.** The same imperfect scan that
  produced the corpus statistics produced these labels, and the heuristic structurally cannot
  see three-column pages — it labelled the back index `single_column` before the index was
  excluded. Column ground truth for these 40 pages, if needed, must be recorded by eye.

## Status

Images: **done** (40/40). Transcriptions in `../labels.jsonl`: **not yet authored** — until they
exist, no OCR number can be quoted from this slice.
