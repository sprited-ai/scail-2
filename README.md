# scail-2

> **Unofficial community packaging.** This is not our model and we are not
> affiliated with its authors. We packaged
> [SCAIL-2](https://github.com/zai-org/SCAIL-2) (Zhipu AI — code Apache-2.0,
> weights MIT) and its official ComfyUI implementation for
> [Replicate](https://replicate.com); **we earn nothing** — compute fees go to
> Replicate. Authors who want this changed or taken down:
> [open an issue](https://github.com/sprited-ai/scail-2/issues) and we comply
> immediately.

**[SCAIL-2](https://arxiv.org/abs/2606.10804) on [Replicate](https://replicate.com):
end-to-end character animation.** Give it a picture of a character and a
video of someone (or something) moving, and it makes the character perform
that motion — no skeleton extraction, no pose maps. It handles humans, animals,
robots, hand-drawn and pixel-art characters, and 3D actions like spins and
flips that pose-driven models (Wan-Animate) break on.

What this endpoint gives you:

- **Two modes** — `animation` (your character, on its own background, does the
  motion) and `replacement` (your character is dropped into the driving video,
  keeping its scene and lighting).
- **Masks in, masks out** — bring your own reference / per-frame driving masks
  (grayscale mattes or SCAIL-2 palette colours), or let `auto_mask` derive them
  from the image's alpha channel or [BiRefNet](https://github.com/ZhengPeng7/BiRefNet).
  `return_masks` hands them back so you can iterate.
- **Fast or quality** — the default is the official ComfyUI recipe (lightx2v
  step/CFG-distill LoRA + DPO LoRA, 6 steps, CFG 1: 81 frames at 512p in
  about a minute); `quality` is the paper's sampler (UniPC, 40 steps, CFG 5),
  about 12x slower. Steps / CFG / shift are exposed.
- **Official LoRAs** — the Bias-Aware DPO LoRA and the relighting LoRA (for
  replacement mode) ship in the image, dialled in by strength.
- **Your frame rate, held not interpolated** — resample the driving video to
  any fps (sprite-sheet timing survives), output at the same rate. Up to 161
  frames per call, generated in overlapping windows exactly like the official
  ComfyUI "extend" workflow.
- **Same numbers as ComfyUI** — this runs ComfyUI's own `WanSCAILToVideo`
  node (v0.33.1, pinned) on the Comfy-Org fp8 repack, so a local ComfyUI
  workflow with the same inputs, seed and settings reproduces the result.

## Inputs

| input | default | notes |
|---|---|---|
| `image` | — | reference character. Transparent PNGs are flattened on white and their alpha becomes the reference mask |
| `video` | — | driving video; its aspect ratio sets the output size |
| `prompt` | `""` | describe the character and the motion (a description of the final video, not instructions) |
| `negative_prompt` | `""` | e.g. `distorted limbs, camera movement`; only matters with CFG > 1 (`quality`). Wan's stock negative prompt is *not* applied: its painting/style terms push stylized characters toward 3D-CG |
| `mode` | `animation` | `animation` or `replacement` |
| `image_mask` / `video_mask` | — | optional masks (see below) |
| `auto_mask` | `true` | derive missing masks (alpha channel → BiRefNet). Off = no masks |
| `resolution` | `512p` | short side 512 or 704; aspect follows the driving video (16:9 → 896x512, square → 512x512) |
| `width` / `height` | `0` | explicit size (multiples of 32) — centre-crops the driving video to that aspect |
| `num_frames` | `0` | 0 = all driving frames (max 161); rounded to 4k+1 |
| `fps` | `0` | resample the driving video to this rate first (nearest frame, no interpolation); 0 = source rate |
| `preset` | `fast` | `fast` = lightx2v LoRA 0.8, Euler 6 steps / CFG 1 / shift 5 (official ComfyUI recipe); `quality` = UniPC 40 steps / CFG 5 / shift 3 (paper) |
| `steps`, `guidance_scale`, `shift` | `0` | override the preset (0 = preset default) |
| `dpo_lora` | `1.0` | strength of the official Bias-Aware DPO LoRA (0 = off; the released base checkpoint is pre-DPO) |
| `relight_lora` | `0` | strength of the official relighting LoRA (replacement mode) |
| `pose_strength` | `1.0` | weight of the motion conditioning |
| `seed` | random | |
| `return_masks` | `false` | also return the reference mask (PNG) and driving mask (MP4) that were used |

Output: `video` (H.264 MP4 at the driving frame rate), `seed`, and optionally
`reference_mask` / `driving_mask`.

### Masks

SCAIL-2 binds the character in the reference image to the moving subject in
the driving video through colour-coded masks (the model was trained on a fixed
palette; the first identity is pure blue):

| | driving-video mask background | reference mask background |
|---|---|---|
| `animation` | black — the driving video's background is *not* shown | white — the reference's background *is* shown |
| `replacement` | white — the driving video's background is kept | black |

You can pass either a plain matte (white = subject) — it is coloured and put on
the right background for the mode — or a mask already in SCAIL-2 colours,
which passes through untouched (multi-identity masks work this way). With
`auto_mask` on and no mask given, a transparent reference uses its alpha;
everything else is matted with BiRefNet (single subject). Masks are optional:
single-character animation works without them, they mostly help identity
binding and multi-character scenes.

### Runtime

Replicate stops predictions at 30 minutes, so requests whose estimated sampling
time exceeds ~26 minutes are refused up front with a suggestion (fewer frames,
smaller resolution, or `fast`). Measured on an RTX PRO 6000 (H100 is expected to
be similar or faster):

| preset | frames × size | sampling |
|---|---|---|
| fast | 81 × 896x512 | 85 s |
| fast | 161 × 512x512 (3 windows) | 69 s |
| quality | 33 × 512x512 | 143 s |
| quality | 81 × 896x512 | 1043 s |

### Long videos

Up to 81 frames is one pass. Longer requests are split into equal windows
(≤ 81 frames, 4k+1) chained through the node's `previous_frames` anchor with
the trained 5-frame overlap, the same construction as the official ComfyUI
extend workflow. 157 frames = 2 windows of 81; 161 = 3 windows of 57.

## Compared with fal's `fal-ai/scail-2`

fal exposes prompt / image / video / mode / resolution / steps / CFG / shift /
seed, plus `driving_type` (end-to-end or pose) and `subject_type` (human /
animal, which drives its SAM3 masking). This endpoint adds user-supplied and
returned masks, fps control with frame holding, explicit sizes, the fast
distilled preset, the official DPO / relighting LoRAs and pose strength; it
does not (yet) do skeleton-based `pose` driving or SAM3 text-prompted
multi-person masking (see roadmap).

## Licenses

- SCAIL-2 code: Apache-2.0 (Zhipu AI). Weights: MIT ([zai-org/SCAIL-2](https://huggingface.co/zai-org/SCAIL-2), [Comfy-Org/SCAIL-2](https://huggingface.co/Comfy-Org/SCAIL-2)).
- Wan 2.1 VAE / umT5 text encoder: Apache-2.0. CLIP ViT-H: MIT (open_clip).
- lightx2v distillation LoRA: Apache-2.0. BiRefNet: MIT.
- ComfyUI (runtime, unmodified, run as a separate process): GPL-3.0.
- This packaging: MIT. Outputs are yours; commercial use is fine under all of the above. See [NOTICE](NOTICE).

## Roadmap

- SAM3 text-prompted masks (`subject_type`-style multi-identity tracking) — needs a license review of the SAM 3 weights.
- Skeleton `pose` driving via SCAIL-Pose.
- Multi-reference (extra views / background references) — the node supports it.

## Deploy

```
cog login && cog push r8.im/sprited/scail-2
```

Unit tests for the input preparation and the graph builder (no GPU):

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python av numpy pillow pytest websocket-client
.venv/bin/python -m pytest tests
```

Packaged by [Sprited](https://spritedx.com).

## Citation

```bibtex
@misc{yan2026scail2,
  title={SCAIL-2: Unifying Controlled Character Animation with End-to-end In-Context Conditioning},
  author={Yan, Wenhao and Guo, Fengjia and Yang, Zhuoyi and Tang, Jie},
  year={2026},
  eprint={2606.10804},
  archivePrefix={arXiv}
}
```
# Runtime limits

The predictor worker has a 10-minute setup deadline and a 14-minute prediction
deadline. The latter covers preprocessing, sampling and output encoding. On a
deadline it stops its ComfyUI subprocess and exits with code 124. ComfyUI graph
execution also has a 12-minute budget; connection failures terminate the server.

These limits start inside the worker. They do **not** bound platform scheduling,
image pulls or anything before Python starts. API callers must send
`Cancel-After: 15m` when creating a prediction. `tools/replicate_predict.sh` sends
this header, bounds polling, and requests cancellation on abnormal exit. It does
not retry prediction creation. Playground requests do not automatically inherit
the API header.

## Bundled weights

`cog build` downloads all seven SCAIL-2 weights into the image, alongside the
BiRefNet cache. Startup validates their sizes and fails clearly if any file is
missing or truncated; it does not download replacement weights. The image is
approximately 50 GB uncompressed. Replicate still needs to pull the image on a
cold boot, but model setup no longer depends on Hugging Face downloads.

Replicate updates runtime dependencies before calling `setup()`. The image
removes the uv-managed interpreter's `EXTERNALLY-MANAGED` marker during build
so that bootstrap can install its dependencies. This affects only the isolated
container interpreter, not the build host or a user's Python environment.

The final runtime is pinned to Cog/Coglet 0.23.0, matching the builder used for
the hosted validation. That image completed setup on an H100 in 9.4 seconds.
An earlier five-minute test spent 286 seconds before processing and reached
model loading before its deadline. The smaller-image version below subsequently
passed hosted E2E with a ten-minute request deadline. Image pull and GPU allocation
remain possible contributors to startup delay; the available logs do not distinguish them.

### Smaller-image validation

The hosted E2E-validated version `b7d7a4fb451d4e0e6d24b20b126e927a3f0854edb193f2b9513aaa0b0f86079a`
uses `cog.python-base.yaml`. Reproduce it with Cog 0.23.0:

```sh
cog build -f cog.python-base.yaml --use-cuda-base-image=false -t scail2-python-base:test
cog push -f cog.python-base.yaml --use-cuda-base-image=false r8.im/sprited/scail-2
```

It includes the same weights and is 39.12 GB uncompressed, compared with
49.60 GB for the CUDA-base variant. On gin it passed the same 33-frame test
with network disabled and no model mounts: readiness 6.92 seconds, prediction
40.78 seconds. These are single-run timings, not an isolated performance benchmark.

Hosted request `vaz9pxr1qdrp40d0xwcb8y2dhm` was created at
2026-09-29T21:35:23Z and aborted at 21:40:23Z under `Cancel-After: 5m`.
It never entered processing and returned no logs or output. Reducing the image
size has therefore not yet demonstrated a solution to the hosted startup delay.

A subsequent request, `5v9wzfjybdrne0d0xxdas77gfc`, succeeded on the same version
with `Cancel-After: 10m`: 378.02 seconds before processing, 29.99 seconds of
inference, and 408.01 seconds total. Its 896×512, 33-frame MP4 was retrieved and
an output frame inspected. This verifies the fast single-subject test, not all
input combinations or consistent cold-start latency. No automatic retry was used.
