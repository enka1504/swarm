#!/bin/bash
# Benchmark 60 seeds (10 per region) and generate depth videos for each seed
#
# Usage: bash scripts/bench_and_video.sh [submission_zip]
#   Default submission: Submission/submission.zip

set -e

SUBMISSION="${1:-Submission/submission.zip}"
SEED_FILE="/tmp/bench_60_seeds.json"
VIDEO_DIR="/root/swarm/videos"
WORKERS=12

echo "=== Step 1: Benchmark 60 seeds (10 per region) ==="
echo "Submission: $SUBMISSION"
echo "Seed file:  $SEED_FILE"
echo ""

python -m swarm.cli benchmark \
    --model "$SUBMISSION" \
    --seeds-per-group 10 \
    --workers "$WORKERS" \
    --relax-timeouts \
    --save-seed-file "$SEED_FILE"

echo ""
echo "=== Step 2: Generate depth videos ==="
echo "Output dir: $VIDEO_DIR"
echo ""

mkdir -p "$VIDEO_DIR"

python -m swarm.cli video \
    --model "$SUBMISSION" \
    --seed-file "$SEED_FILE" \
    --mode depth \
    --out "$VIDEO_DIR" \
    --width 640 \
    --height 480 \
    --fps 25

echo ""
echo "=== Done ==="
echo "Videos saved to: $VIDEO_DIR"
echo "Seeds saved to:  $SEED_FILE"
