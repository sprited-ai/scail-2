#!/bin/bash
# Run the predictor on gin against its ComfyUI checkout (no docker). Usage:
#   tools/gin_run.sh REF VIDEO OUT [extra run_local args...] [< jobs.jsonl with --serve]
set -euo pipefail
cd /mnt/stash/scail2-dev/repo
export COMFY_DIR=/home/gin/dev/ComfyUI COMFY_PORT=8199 SCAIL2_WORK=/mnt/stash/scail2-dev/work
export PYTHONPATH=/mnt/stash/scail2-dev/pylib
REF=$1; VID=$2; OUT=$3; shift 3
exec /home/gin/dev/ComfyUI/venv/bin/python tools/run_local.py --image "$REF" --video "$VID" --out "$OUT" "$@"
