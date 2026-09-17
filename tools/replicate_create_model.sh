#!/bin/bash
# Create the public model entry on Replicate before the first `cog push`.
# Reads REPLICATE_API_TOKEN from the environment (or ../sprute/.env).
set -euo pipefail
: "${REPLICATE_API_TOKEN:=$(grep -E '^REPLICATE_API_TOKEN=' "$(dirname "$0")/../../sprute/.env" | cut -d= -f2- | tr -d '"'"'"')}"
OWNER=${OWNER:-sprited}; NAME=${NAME:-scail-2}; HARDWARE=${HARDWARE:-gpu-h100}
curl -sS -X POST https://api.replicate.com/v1/models \
  -H "Authorization: Bearer $REPLICATE_API_TOKEN" -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "owner": "$OWNER",
  "name": "$NAME",
  "visibility": "public",
  "hardware": "$HARDWARE",
  "description": "SCAIL-2 end-to-end character animation (zai-org): animate a character image with any driving video, no skeleton extraction. Animation + replacement modes, own or automatic masks, official DPO/relight LoRAs, quality and fast presets. Apache-2.0 / MIT.",
  "github_url": "https://github.com/sprited-ai/scail-2",
  "paper_url": "https://arxiv.org/abs/2606.10804",
  "license_url": "https://huggingface.co/zai-org/SCAIL-2"
}
JSON
echo
