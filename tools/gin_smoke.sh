#!/bin/bash
# CPU smoke test of a built cog image: start the cog HTTP server with the GPU
# hidden, wait for setup() to finish (ComfyUI boots, node check passes) and
# print the health status + logs. Usage: tools/gin_smoke.sh IMAGE [timeout_s]
set -uo pipefail
IMAGE=$1; TIMEOUT=${2:-900}
NAME=scail2-smoke-$$
docker run -d --name "$NAME" -e CUDA_VISIBLE_DEVICES= -p 127.0.0.1:5055:5000 "$IMAGE" >/dev/null
for i in $(seq 1 $((TIMEOUT / 5))); do
  H=$(curl -s --max-time 5 http://127.0.0.1:5055/health-check || true)
  S=$(echo "$H" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("status"))' 2>/dev/null)
  case "$S" in READY|SETUP_FAILED|DEFUNCT) break ;; esac
  if ! docker ps -q --filter "name=$NAME" | grep -q .; then S=EXITED; break; fi
  sleep 5
done
echo "health: ${S:-timeout} after $((i * 5))s"
echo "$H" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get("setup"), indent=1)[:2000])' 2>/dev/null
echo "=== container logs (tail) ==="; docker logs "$NAME" 2>&1 | tail -40
docker rm -f "$NAME" >/dev/null 2>&1
[ "${S:-}" = "READY" ]
