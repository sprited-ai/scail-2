"""Run the predictor outside Replicate (e.g. on gin) against an existing
ComfyUI checkout. Usage:

  COMFY_DIR=/home/gin/dev/ComfyUI COMFY_PORT=8199 SCAIL2_WORK=/mnt/stash/scail2-dev/work \
  python tools/run_local.py --image ref.png --video drive.mp4 --preset fast --num_frames 33 --out out.mp4 [--serve]

--serve keeps the ComfyUI server up and reads further JSON job lines from stdin
(one per line, same keys as the CLI flags) so the 14B model loads only once.
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))        # cog shim
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cog_shim  # noqa: E402
sys.modules["cog"] = cog_shim

import predict  # noqa: E402

if os.environ.get("COMFY_PORT"):
    _orig = predict.ComfyServer.__init__
    def _init(self, *a, **kw):
        kw["port"] = int(os.environ["COMFY_PORT"])
        _orig(self, *a, **kw)
    predict.ComfyServer.__init__ = _init


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default=""); ap.add_argument("--negative_prompt", default="")
    ap.add_argument("--mode", default="animation"); ap.add_argument("--image_mask"); ap.add_argument("--video_mask")
    ap.add_argument("--auto_mask", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--resolution", default="512p"); ap.add_argument("--width", type=int, default=0); ap.add_argument("--height", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=0); ap.add_argument("--fps", type=int, default=0)
    ap.add_argument("--preset", default="quality"); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--guidance_scale", type=float, default=0); ap.add_argument("--shift", type=float, default=0)
    ap.add_argument("--dpo_lora", type=float, default=0); ap.add_argument("--relight_lora", type=float, default=0)
    ap.add_argument("--pose_strength", type=float, default=1.0); ap.add_argument("--seed", type=int)
    ap.add_argument("--return_masks", type=lambda s: s.lower() != "false", default=False)
    ap.add_argument("--out", required=True); ap.add_argument("--serve", action="store_true")
    return ap.parse_args(argv)


def run_job(pred, a):
    kw = {k: (predict.Path(v) if k in ("image", "video", "image_mask", "video_mask") and v else v)
          for k, v in vars(a).items() if k not in ("out", "serve")}
    out = pred.predict(**kw)
    shutil.copy(out.video, a.out)
    print(json.dumps({"out": a.out, "seed": out.seed,
                      "reference_mask": str(out.reference_mask) if out.reference_mask else None,
                      "driving_mask": str(out.driving_mask) if out.driving_mask else None}), flush=True)


if __name__ == "__main__":
    a = parse()
    pred = predict.Predictor()
    pred.setup()
    try:
        run_job(pred, a)
        if a.serve:
            print("[serve] send JSON job lines on stdin", flush=True)
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                job = json.loads(line)
                argv = []
                for k, v in job.items():
                    argv += [f"--{k}", str(v)]
                try:
                    run_job(pred, parse(argv))
                except Exception as e:  # keep serving
                    print(json.dumps({"error": repr(e)}), flush=True)
    finally:
        pred.comfy.stop()
