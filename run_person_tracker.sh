#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo " YOLOv8m + ByteTrack Advanced Person Tracker"
echo " - Draw ROI zones per machine (drag on first frame)"
echo " - Source/Dest box piece-work tracking (1 unit/cycle)"
echo " - Hand / wrist motion Working & Idle tracking"
echo " - Crowding alert (>1 person per zone)"
echo " - Re-ID buffer (3s ghost window)"
echo " - Automated Gemma 3 Industrial Efficiency & OEE Report"
echo "============================================================"
echo ""

PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" person_tracker.py \
    --video "person video/Cam_192.168.170.64_2026-09-25_10-53-57.avi" \
    --conf 0.25 \
    --ghost-sec 3 \
    --idle-sec 10 \
    --work-buffer 3 \
    --out-fps 0 \
    --gemma-model "gemma3:12b"

echo ""
echo "[Done] Process finished."
