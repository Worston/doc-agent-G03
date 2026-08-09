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
seed: 42`. `select_heldout.py` in this directory IS that rule — running it reproduces exactly the
id list below, and a grader can audit the selection without taking this README's word for it.

**1. Section-level split first.** A1 committed to *"Split unit = SECTION (the book's own
Class/Part divisions), not page. Approximately 70% train / 15% validation / 15% test by section,
assigned whole."* Sections are recovered from the running heads — `CLASS n` inside the recipe
part (pp. 219–423), the Roman-numeral `Section N` elsewhere, `SUPPLEMENT` for the wartime front
supplement. Three wrinkles had to be handled: verso and recto carry **different** running heads
(verso repeats the section, recto names the class), so heads are forward-filled; the class digits
arrive space-split from OCR, e.g. `CLASS 1 6— VEGETABLES` is Class 16 — which is the very
boundary A1 cites as its example; and section **opener** pages carry no running head at all, so
they need a separate rule (see *Correction* below — getting this wrong leaked train pages into
the first cut of this slice). Realised on 491 content pages:

| split | content pages | share | sections |
|---|---|---|---|
| train | 335 | 68.2% | 24 |
| validation | 80 | 16.3% | 6 |
| test | 76 | 15.5% | 8 |

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
| two-column share (heuristic) | 35.0% (14) | 33.8% (166) |
| recipe-class pages | 42% (17) | 36.0% |
| median words/page | 680 | 641 |
| word range | 94 – 1,038 | 86 – 1,064 |
| median speckles/megapixel | 83.5 | 61.7 |
| speckle p10 / p90 | 20.0 / 131.9 | 10.5 / 131.3 |

Speckles/megapixel counts connected components of <= 4 px after Otsu binarisation — a proxy for
dust, bleed-through and scan dirt. It is reported because *degraded scans* is our secondary data
condition, and a held-out set that happened to be clean would flatter the OCR score. It does not:
the slice sits above the corpus median (83.5 vs 61.7), so if anything it is the harder end.

The word range now reaches down to 94 because p138 is a genuinely near-empty page — two short
mending paragraphs and a printed "(Paste or Write Here / Scraps or Memos. / of Your Own)"
invitation, the rest blank. It is kept deliberately: an OCR stage that hallucinates text into
white space fails visibly on it.

## The 40 pages

`cross%` is the share of word boxes straddling the page centre-line, and `heuristic` is what
A1's rule makes of it (< 1.2% ⇒ two-column). `verified layout` was recorded **by eye** from the
300 DPI image while transcribing; ⚠ marks the 25 pages where the two disagree. `words (oracle)`
counts the transcription in `../labels.jsonl`, not the PDF text layer, so it differs slightly
from the A1 word counts.

| page_id | pdf page | section | words (oracle) | cross% | heuristic | verified layout |
|---|---|---|---|---|---|---|
| hkb_p0124 | 124 | Section II | 773 | 2.03 | single_column | mixed ⚠ |
| hkb_p0125 | 125 | Section II | 906 | 0.76 | two_column | two_column |
| hkb_p0127 | 127 | Section II | 818 | 4.46 | single_column | mixed ⚠ |
| hkb_p0131 | 131 | Section II | 519 | 4.94 | single_column | mixed ⚠ |
| hkb_p0134 | 134 | Section II | 665 | 3.38 | single_column | mixed ⚠ |
| hkb_p0135 | 135 | Section II | 851 | 4.71 | single_column | two_column ⚠ |
| hkb_p0138 | 138 | Section II | 88 | 3.19 | single_column | two_column ⚠ |
| hkb_p0220 | 220 | CLASS 01 | 899 | 0.11 | two_column | two_column |
| hkb_p0224 | 224 | CLASS 01 | 383 | 6.07 | single_column | single_column |
| hkb_p0253 | 253 | CLASS 07 | 977 | 0.10 | two_column | two_column |
| hkb_p0255 | 255 | CLASS 07 | 391 | 0.49 | two_column | mixed ⚠ |
| hkb_p0257 | 257 | CLASS 07 | 665 | 4.49 | single_column | mixed ⚠ |
| hkb_p0258 | 258 | CLASS 07 | 457 | 0.21 | two_column | two_column |
| hkb_p0260 | 260 | CLASS 07 | 614 | 3.68 | single_column | single_column |
| hkb_p0276 | 276 | CLASS 12 | 928 | 0.10 | two_column | two_column |
| hkb_p0281 | 281 | CLASS 14 | 908 | 4.19 | single_column | two_column ⚠ |
| hkb_p0283 | 283 | CLASS 14 | 516 | 4.58 | single_column | two_column ⚠ |
| hkb_p0286 | 286 | CLASS 14 | 651 | 3.31 | single_column | mixed ⚠ |
| hkb_p0327 | 327 | CLASS 19 | 911 | 4.46 | single_column | two_column ⚠ |
| hkb_p0331 | 331 | CLASS 20 | 840 | 4.60 | single_column | two_column ⚠ |
| hkb_p0334 | 334 | CLASS 20 | 158 | 0.60 | two_column | mixed ⚠ |
| hkb_p0337 | 337 | CLASS 20 | 753 | 1.53 | single_column | mixed ⚠ |
| hkb_p0338 | 338 | CLASS 20 | 963 | 0.10 | two_column | two_column |
| hkb_p0341 | 341 | CLASS 20 | 764 | 0.37 | two_column | mixed ⚠ |
| hkb_p0471 | 471 | Section VII | 357 | 5.25 | single_column | single_column |
| hkb_p0474 | 474 | Section VII | 815 | 1.08 | two_column | mixed ⚠ |
| hkb_p0476 | 476 | Section VII | 573 | 2.00 | single_column | mixed ⚠ |
| hkb_p0477 | 477 | Section VII | 611 | 4.56 | single_column | single_column |
| hkb_p0479 | 479 | Section VII | 172 | 4.55 | single_column | single_column |
| hkb_p0482 | 482 | Section VII | 1013 | 2.12 | single_column | two_column ⚠ |
| hkb_p0484 | 484 | Section VII | 967 | 4.54 | single_column | two_column ⚠ |
| hkb_p0485 | 485 | Section VII | 544 | 0.54 | two_column | two_column |
| hkb_p0487 | 487 | Section VII | 576 | 4.38 | single_column | mixed ⚠ |
| hkb_p0489 | 489 | Section VII | 904 | 4.02 | single_column | two_column ⚠ |
| hkb_p0492 | 492 | Section VII | 626 | 0.15 | two_column | two_column |
| hkb_p0494 | 494 | Section VII | 552 | 2.20 | single_column | mixed ⚠ |
| hkb_p0495 | 495 | Section VII | 426 | 0.23 | two_column | two_column |
| hkb_p0497 | 497 | Section VII | 562 | 4.72 | single_column | single_column |
| hkb_p0501 | 501 | Section VII | 533 | 2.03 | single_column | mixed ⚠ |
| hkb_p0503 | 503 | Section VII | 768 | 0.00 | two_column | mixed ⚠ |

p482 is one of the three pages A1 names as worst-case hyphenation (44 line-break hyphens), so
the slice deliberately contains a known-hard page rather than avoiding it.

The oracle preserves the book's own printing errors verbatim — `amaemic`, `diarrhoe`,
`univerasl`, `other friut`, `disfection`, `formaldeheyde`, `daintly`, `rythmic`, `scrapeing`,
`Mix rdy bread crumbs`. An OCR stage that "helpfully" corrects these will score *worse*, which
is intended: the task is transcription, not editing.

## Honest limits of this split

- **Same book, same typeface, same scanning session.** A1 states this openly: the assignment
  asks for a split by DOCUMENT and we have one document, so section is the strictest correlation
  unit available. No single-corpus split can measure generalisation to an unseen press or
  scanner. The number from this slice is *in-domain* OCR accuracy and must be described so.
- **Section boundaries are themselves OCR-derived.** They come from running heads, which are
  damaged on part of the corpus; heads were directly readable on 330 pages and forward-filled
  elsewhere. A misplaced boundary would move a page between splits — and one did; see
  *Correction* below.
- **The A1 layout heuristic is not usable as layout ground truth.** Measured against the
  eye-verified column above, it agrees on **15 of 40** pages and finds only 14 multi-column
  pages where there are 34. Two causes: it has no *mixed* class at all, and a full-width heading
  or illustration band puts enough word boxes across the centre-line to mask a two-column body.
  This is a **measurement of the A1 rule, not a restatement of it**; A1's reported corpus figure
  (33.0% two-column) is left exactly as submitted, and this slice is the evidence that the A2
  layout stage (`vision/layout.py`) has to do better than a global centre-crossing test.

## Correction — section map v2 (this changes which pages are in the slice)

The first cut of this slice read each page's section from its **first line only**. Section
*opener* pages have no running head — the title is set inside a decorative masthead — so an
opener inherited the previous section by forward-fill, and openers that do carry text print it
in caps with OCR-mangled numerals (`SECTION VIU.` for VIII, `Section 111.` for III). Eight pages
were mis-assigned. Three of them had been selected, and two of those were **not test pages at
all**: p141 and p139 are Section III, p507 is Section VIII — all train sections. That silently
broke the "never trained or tuned on" guarantee this whole directory exists to provide.

`select_heldout.py` (v2, in this directory) adds an opener rule anchored at end-of-line, so that
in-body cross-references such as p486's `... see SECTION V — CONVENIENCES ...` do not match. p139
needs no regex at all — its title is *drawn*, with no text layer — so it is pinned by eye in
`task.yaml → heldout.eye_verified_sections`. The corrected map moves 8 pages, leaves the split
shares within a point of A1 (train 68.2% / val 16.3% / test 15.5% on 491 content pages), and the
redraw replaced 10 of the 40. `test_sections`, `val_sections`, `seed` and the draw rule are
unchanged — only the page→section map was wrong.

Re-derive and audit at any time:

    python3 grading_kit/heldout_pages/select_heldout.py

## Status

Images: **done** (40/40, byte-identical to `data/raw/`).
Transcriptions in `../labels.jsonl`: **done** (40/40, 26,397 words, 0 `[?]` markers).

**Provenance — read before quoting any number.** These transcriptions were drafted by a
vision-language model reading the 300 DPI page images at high zoom, then **checked page by page
against the images by a member of the team**, who corrected them in two passes (commits
`bb437cf`, `d961f71`). Both stages are recorded in each record's `transcription_source` field.
They are deliberately **not** derived from the Internet Archive PDF text layer, which is ABBYY
OCR — scoring OCR against another OCR engine's output would make the metric circular. With human
verification complete, this slice is the reference used for the Section 5 OCR accuracy figure.

Printers' errors and archaic spellings are preserved verbatim, so an OCR stage that silently
"corrects" the page scores *worse*, as intended.
