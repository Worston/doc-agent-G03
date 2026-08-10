# Corpus Provenance

## Source

| Field | Value |
|---|---|
| **Title** | *The Home-Keeping Book* |
| **Author** | Alice Van Leer Carrick (Brown, 1918) |
| **Internet Archive identifier** | `homekeepingbook00brow` |
| **Download URL** | <https://archive.org/details/homekeepingbook00brow> |
| **Download command** | `bash scripts/get_data.sh` (renders PDF → grayscale PNG at 300 DPI) |

## Licence / Usage Rights

Public domain (United States).  Published 1918, well before the 1929 cut-off for US public-domain works.  No restrictions on use, redistribution, or derivative works.  No personal data in the text.

## Page and Word Counts

| Metric | A1 reported | Source |
|---|---|---|
| Total pages (PDF) | 624 | `configs/task.yaml → a1_reported.total_pages` |
| Content pages (≥ 80 words) | 506 | `configs/task.yaml → a1_reported.content_pages` |
| Sparse / blank / plate pages | 118 | `configs/task.yaml → a1_reported.sparse_pages` |
| Total words (content pages) | 318,259 | `configs/task.yaml → a1_reported.total_words` |
| Median words per content page | 626 | `configs/task.yaml → a1_reported.median_content_words` |
| Max words per content page | 1,064 | `configs/task.yaml → a1_reported.max_content_words` |
| Two-column pages | 167 (33 %) | `configs/task.yaml → a1_reported.two_column_pages` |

> Both floors are cleared: **≥ 300 pages** (506 content pages) and **≥ 60,000 words** (318,259 words).

## Size on Disk

| Asset | Size |
|---|---|
| Source PDF (`data/interim/homekeepingbook00brow.pdf`) | ≈ 250 MB |
| 624 grayscale PNG page-images at 300 DPI (`data/raw/`) | ≈ 1.1 GB |

## Scan / Script Difficulty Notes

**Primary data speciality — multi-column reading order.**
Roughly one-third of the content pages (167 / 506 ≈ 33 %) are typeset in two narrow columns separated by a gutter of 25–98 px at 300 DPI.  A naïve top-to-bottom OCR sweep interleaves the columns; the layout detector (`vision/layout.py`, XY-cut) must separate them before transcription.

**Secondary condition — degraded / dirty scans.**
The 1918 scan carries heavy ink bleed-through, foxing spots, and uneven illumination.  Classical preprocessing (deskew + median denoise, `ingest/preprocess.py`) reduces Tesseract CER from 0.091 → 0.080.  The optional diffusion enhancer (`ingest/enhance.py`) is wired but trained in A2.

**Script / language:** English only; no non-Latin scripts.

**Font / typography diversity:** period serif (old-style figures, long-s ligatures absent but ornamental chapter headers present).

## Split Policy

Split unit = **section** (the book's own Class / Section divisions), not individual page.  Every page of a section goes to exactly one split.  This prevents the same physical text appearing in both train and test/val.

| Split | Sections | Pages (approx.) | Fraction |
|---|---|---|---|
| Train | all sections not listed below | ~343 pages | 67.8 % |
| Validation | CLASS 05, CLASS 28, CLASS 30, Section VI, Section IX, Section XI | ~81 pages | 16.1 % |
| Test | CLASS 01, CLASS 07, CLASS 12, CLASS 14, CLASS 19, CLASS 20, Section II, Section VII | ~81 pages | 16.1 % |

Section boundaries are read from running heads (forward-filled, because verso and recto carry different heads).  The exact section list is frozen in `configs/task.yaml → splits`.

## Held-Out Oracle Slice

40 content pages drawn from the test split, stratified by layout type to match the corpus-level two-column share (≈ 14 two-column, ≈ 26 single/mixed).  These pages are in `grading_kit/heldout_pages/`; their ground-truth transcriptions are in `grading_kit/labels.jsonl`.  OCR is **never** trained or tuned on these pages.

Selection is deterministic from `configs/task.yaml → heldout` plus `configs/config.yaml → seed (42)` and can be re-derived and audited at any time.

