#!/bin/bash
# Run one prediction against sprited/scail-2 on Replicate and report timing.
# Usage: tools/replicate_predict.sh IMAGE VIDEO OUT.mp4 '{"preset":"fast","num_frames":81,...}'
# Reads REPLICATE_API_TOKEN from the environment or ../sprute/.env.
set -euo pipefail
# token: env var, else ./.env in the repo root, else ../sprute/.env
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for f in "$ROOT/.env" "$ROOT/../sprute/.env"; do
  [ -n "${REPLICATE_API_TOKEN:-}" ] && break
  [ -f "$f" ] && REPLICATE_API_TOKEN=$(grep -E '^REPLICATE_API_TOKEN=' "$f" | tail -1 | sed -e 's/^[^=]*=//' -e "s/[\"']//g")
done
export REPLICATE_API_TOKEN
MODEL=${MODEL:-sprited/scail-2}
IMAGE=$1; VIDEO=$2; OUT=$3; EXTRA=${4:-'{}'}
AUTH="Authorization: Bearer $REPLICATE_API_TOKEN"
API=https://api.replicate.com/v1

upload() { curl -sS -X POST "$API/files" -H "$AUTH" -F "content=@$1" | python3 -c 'import sys,json; print(json.load(sys.stdin)["urls"]["get"])'; }
IMG_URL=$(upload "$IMAGE"); VID_URL=$(upload "$VIDEO")
echo "uploaded: $IMG_URL"; echo "uploaded: $VID_URL"

BODY=$(python3 - "$IMG_URL" "$VID_URL" "$EXTRA" <<'PY'
import json, sys
inp = {"image": sys.argv[1], "video": sys.argv[2]}
inp.update(json.loads(sys.argv[3]))
print(json.dumps({"input": inp}))
PY
)
# private models can't use the models/…/predictions shortcut: run the latest version by id
VERSION=${VERSION:-$(curl -sS "$API/models/$MODEL" -H "$AUTH" | python3 -c 'import sys,json; print(json.load(sys.stdin)["latest_version"]["id"])')}
echo "version: $VERSION"
BODY=$(echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); d['version']='$VERSION'; print(json.dumps(d))")
PRED=$(curl -sS -X POST "$API/predictions" -H "$AUTH" -H "Content-Type: application/json" -d "$BODY")
ID=$(echo "$PRED" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("id") or sys.exit("create failed: "+json.dumps(d)))')
echo "prediction: https://replicate.com/p/$ID"

while :; do
  P=$(curl -sS "$API/predictions/$ID" -H "$AUTH")
  STATUS=$(echo "$P" | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  case "$STATUS" in
    succeeded|failed|canceled) break ;;
  esac
  sleep 10
done
echo "$P" | python3 -c '
import sys, json
d = json.load(sys.stdin)
print("status:", d["status"])
print("metrics:", json.dumps(d.get("metrics")))
if d["status"] != "succeeded":
    print("error:", d.get("error")); print((d.get("logs") or "")[-3000:]); sys.exit(1)
out = d["output"]; print("output:", json.dumps(out)[:500])
print((d.get("logs") or "")[-1500:])
open("/tmp/_scail2_out_url", "w").write(out["video"] if isinstance(out, dict) else out)
'
curl -sSL -o "$OUT" "$(cat /tmp/_scail2_out_url)" && echo "saved $OUT" && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "$OUT"
