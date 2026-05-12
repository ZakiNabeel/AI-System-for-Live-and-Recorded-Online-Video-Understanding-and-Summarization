#!/usr/bin/env bash
# =============================================================================
# DIP Video Understanding — Demo Reproduction Script
# =============================================================================
# Usage: bash demo/demo_commands.sh
# Run from the project root directory.
#
# Prerequisites:
#   - pip install -r requirements.txt
#   - .env file with ANTHROPIC_API_KEY or GEMINI_API_KEY
#   - ffmpeg installed and on PATH
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# Configuration — EDIT THESE BEFORE RUNNING
# ---------------------------------------------------------------------------
RECORDED_URL="https://www.youtube.com/watch?v=REPLACE_WITH_VIDEO_ID"
DOMAIN="tutorial-strategy"   # Options: education trading medical law tutorial-strategy
ENABLE_CAPTIONS=""            # Set to "--enable-captions" to enable vision captioning
ENABLE_QA=""                  # Set to "--enable-qa" to generate Q&A pairs

echo "=============================================="
echo " DIP Video Understanding System — Demo"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "[OK] Loaded .env"
else
    echo "[WARN] No .env file found. API keys must be set as environment variables."
fi

# ---------------------------------------------------------------------------
# Recorded Video Demo
# ---------------------------------------------------------------------------
echo ""
echo "--- RECORDED VIDEO DEMO ---"
echo "URL:    $RECORDED_URL"
echo "Domain: $DOMAIN"
echo ""

python -m src.pipeline \
    --url "$RECORDED_URL" \
    --mode recorded \
    --domain "$DOMAIN" \
    $ENABLE_CAPTIONS \
    $ENABLE_QA

# Find the latest run_id
RUN_ID=$(ls -t data/output/ 2>/dev/null | head -1)

if [ -z "$RUN_ID" ]; then
    echo "[ERROR] No output found. Pipeline may have failed."
    exit 1
fi

echo ""
echo "Pipeline complete. Run ID: $RUN_ID"
echo "Outputs in: data/output/$RUN_ID/"
echo ""

# ---------------------------------------------------------------------------
# Copy outputs to demo/recorded_demo/
# ---------------------------------------------------------------------------
echo "--- Copying outputs to demo/recorded_demo/ ---"

mkdir -p demo/recorded_demo/frames

for f in summary.md summary.html report.json chapters.txt report_card.md performance_report.json strategy.md strategy.py strategy.json study_notes.md trade_log.csv clinical_notes.md case_brief.md; do
    if [ -f "data/output/$RUN_ID/$f" ]; then
        cp "data/output/$RUN_ID/$f" "demo/recorded_demo/"
        echo "  Copied: $f"
    fi
done

# Copy representative frames (up to 10)
if [ -d "data/frames/$RUN_ID" ]; then
    FRAMES=$(ls data/frames/$RUN_ID/frame_*.png 2>/dev/null | head -10)
    for f in $FRAMES; do
        cp "$f" demo/recorded_demo/frames/
    done
    echo "  Copied frames: $(echo $FRAMES | wc -w)"
fi

# Write video_info.json
python -c "
import json, sys, os
manifest_path = f'data/intermediate/$RUN_ID/manifest.json'
info = {'run_id': '$RUN_ID', 'url': '$RECORDED_URL', 'domain': '$DOMAIN'}
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    info['title'] = m.get('title', '')
    info['duration_sec'] = m.get('duration_sec', 0)
print(json.dumps(info, indent=2))
" > demo/recorded_demo/video_info.json

echo "  Wrote: video_info.json"

echo ""
echo "=============================================="
echo " Recorded demo outputs ready in demo/recorded_demo/"
echo "=============================================="

# ---------------------------------------------------------------------------
# Live Demo Instructions
# ---------------------------------------------------------------------------
echo ""
echo "--- LIVE DEMO ---"
echo "To run the live demo via Streamlit UI:"
echo "  1. streamlit run app.py"
echo "  2. Select Mode = live"
echo "  3. Enter a YouTube Live URL"
echo "  4. Click Start Live Analysis"
echo ""
echo "To run live demo via CLI:"
echo "  python -m src.pipeline_live --url 'LIVE_STREAM_URL'"
echo ""
echo "Take screenshots at:"
echo "  - Before clicking Start           -> demo/live_demo/screenshot_01_start.png"
echo "  - After chunk_count > 0           -> demo/live_demo/screenshot_02_running.png"
echo "  - When rolling summary is visible -> demo/live_demo/screenshot_03_summary.png"
echo "  - After clicking Stop             -> demo/live_demo/screenshot_04_final.png"
