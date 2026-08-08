#!/usr/bin/env bash
# A1/A2 — fetch or recreate the CulinaryHeritageQA corpus into data/raw/.
#
# Source: "The Home-Keeping Book" (Brown, 1918), Internet Archive homekeepingbook00brow.
# Public domain (US, published pre-1929). See data/provenance.md.
#
# The archive PDF stores each page as a 2138x3093 scan at 300 ppi, so we render at
# 300 DPI — the native resolution. Rendering higher only upsamples; lower loses the
# thin strokes Tesseract needs on this period typeface.
#
# Overrides:
#   SOURCE_PDF=/path/to/local.pdf   use a local copy instead of downloading
#   DPI=300  RAW_DIR=data/raw  IA_ID=homekeepingbook00brow
set -euo pipefail

IA_ID="${IA_ID:-homekeepingbook00brow}"
DPI="${DPI:-300}"
RAW_DIR="${RAW_DIR:-data/raw}"
WORK_DIR="${WORK_DIR:-data/interim}"
SOURCE_PDF="${SOURCE_PDF:-$WORK_DIR/$IA_ID.pdf}"
EXPECTED_PAGES="${EXPECTED_PAGES:-624}"

command -v pdftoppm >/dev/null || {
  echo "ERROR: pdftoppm not found. Install poppler (macOS: brew install poppler)." >&2
  exit 1
}

mkdir -p "$RAW_DIR" "$WORK_DIR"

have=$(find "$RAW_DIR" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
if [ "$have" -eq "$EXPECTED_PAGES" ]; then
  echo "corpus already built: $have page images in $RAW_DIR — nothing to do"
  exit 0
fi

if [ ! -f "$SOURCE_PDF" ]; then
  echo "downloading $IA_ID.pdf from archive.org ..."
  curl -fL --retry 3 -o "$SOURCE_PDF" "https://archive.org/download/$IA_ID/$IA_ID.pdf"
fi

echo "rendering $SOURCE_PDF -> $RAW_DIR at ${DPI} DPI (grayscale PNG) ..."
pdftoppm -r "$DPI" -gray -png "$SOURCE_PDF" "$RAW_DIR/$IA_ID"

got=$(find "$RAW_DIR" -maxdepth 1 -name '*.png' | wc -l | tr -d ' ')
echo "wrote $got page images to $RAW_DIR"
[ "$got" -eq "$EXPECTED_PAGES" ] || {
  echo "ERROR: expected $EXPECTED_PAGES pages, got $got" >&2
  exit 1
}
