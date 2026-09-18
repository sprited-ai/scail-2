#!/bin/bash
# End-to-end test of a built cog image on gin WITH the GPU, without the nvidia
# container runtime (gin's NVML is version-mismatched): mount the device nodes
# and the driver's user-space libraries by hand, start the cog HTTP server, wait
# for READY, and run one prediction through cog's own API (data-URI inputs).
# Usage: tools/gin_gpu_e2e.sh IMAGE REF.png DRIVE.mp4 OUT.mp4 '{"preset":"fast",...}'
set -uo pipefail
IMAGE=$1; REF=$2; VID=$3; OUT=$4; EXTRA=${5:-'{}'}
NAME=scail2-e2e-$$; PORT=5056
LIB=/usr/lib/x86_64-linux-gnu; VER=$(ls $LIB/libcuda.so.[0-9]* | sed 's/.*libcuda\.so\.//' | sort -V | tail -1)
docker run -d --name "$NAME" -p 127.0.0.1:$PORT:5000 \
  --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
  -v $LIB/libcuda.so.$VER:$LIB/libcuda.so.1:ro \
  -v $LIB/libnvidia-ptxjitcompiler.so.$VER:$LIB/libnvidia-ptxjitcompiler.so.1:ro \
  -v $LIB/libnvidia-nvvm.so.$VER:$LIB/libnvidia-nvvm.so.4:ro \
  "$IMAGE" >/dev/null || exit 1
t0=$(date +%s)
for i in $(seq 1 180); do
  H=$(curl -s --max-time 5 http://127.0.0.1:$PORT/health-check || true)
  S=$(echo "$H" | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status"))' 2>/dev/null)
  case "$S" in READY|SETUP_FAILED|DEFUNCT) break ;; esac
  docker ps -q --filter "name=$NAME" | grep -q . || { S=EXITED; break; }
  sleep 5
done
echo "health: ${S:-timeout} after $(( $(date +%s) - t0 ))s"
if [ "${S:-}" != "READY" ]; then docker logs "$NAME" 2>&1 | tail -40; docker rm -f "$NAME" >/dev/null; exit 1; fi
BODY=$(python3 - "$REF" "$VID" "$EXTRA" <<'PY'
import base64, json, mimetypes, sys
def uri(p):
    mt = mimetypes.guess_type(p)[0] or "application/octet-stream"
    return f"data:{mt};base64," + base64.b64encode(open(p, "rb").read()).decode()
inp = {"image": uri(sys.argv[1]), "video": uri(sys.argv[2])}
inp.update(json.loads(sys.argv[3]))
print(json.dumps({"input": inp}))
PY
)
t1=$(date +%s)
echo "$BODY" | curl -s --max-time 3600 -X POST http://127.0.0.1:$PORT/predictions -H "Content-Type: application/json" -d @- > /tmp/e2e_resp.json
echo "predict wall: $(( $(date +%s) - t1 ))s"
python3 - /tmp/e2e_resp.json "$OUT" <<'PY'
import base64, json, re, sys
d = json.load(open(sys.argv[1]))
print("status:", d.get("status"), "| metrics:", d.get("metrics")); print("error:", d.get("error"))
logs = [l for l in (d.get("logs") or "").splitlines() if not re.match(r"^\[comfy\] .* step ", l)]
print("\n".join(logs[-30:]))
out = d.get("output") or {}
v = out.get("video") if isinstance(out, dict) else out
if v and v.startswith("data:"):
    open(sys.argv[2], "wb").write(base64.b64decode(v.split(",", 1)[1])); print("saved", sys.argv[2])
PY
docker rm -f "$NAME" >/dev/null 2>&1
[ -s "$OUT" ] && ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 "$OUT"
