# scail-2

> **Unofficial community packaging.** This is not our model and we are not
> affiliated with its authors. We packaged
> [SCAIL-2](https://github.com/zai-org/SCAIL-2) (Zhipu AI — code Apache-2.0,
> weights MIT) and its official ComfyUI implementation for
> [Replicate](https://replicate.com); **we earn nothing** — compute fees go to
> Replicate. Authors who want this changed or taken down:
> [open an issue](https://github.com/sprited-ai/scail-2-on-replicate/issues) and we comply
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
  (grayscale mattes or SCAIL-2 palette colours), or use embedded animated-WebP transparency for driving masks. Otherwise let `auto_mask` derive them
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
  node (v0.33.1, pinned), with the Comfy-Org FP8 scaled checkpoint.
  Matching settings preserves the workflow contract; bit-identical results
  across GPU types and ComfyUI versions are not guaranteed.

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
| `num_frames` | `0` | Maximum frames from the start, after FPS conversion. 0 = available video up to 161 frames. 81 at 24 fps is about 3.4 seconds. Does not extend shorter inputs; trimmed to 5, 9, 13, ... frames. |
| `fps` | `0` | resample the driving video to this rate first (nearest frame, no interpolation); 0 = source rate |
| `preset` | `fast` | `fast` = lightx2v LoRA 0.8, Euler 6 steps / CFG 1 / shift 5 (official ComfyUI recipe); `quality` = UniPC 40 steps / CFG 5 / shift 3 (paper) |
| `steps`, `guidance_scale`, `shift` | `0` | override the preset (0 = preset default) |
| `dpo_lora` | `1.0` | strength of the official Bias-Aware DPO LoRA (0 = off; the released base checkpoint is pre-DPO) |
| `relight_lora` | `0` | strength of the official relighting LoRA (replacement mode) |
| `pose_strength` | `1.0` | weight of the motion conditioning |
| `seed` | random | |
| `return_masks` | `false` | also return the reference mask (PNG) and driving mask (lossless FFV1 MKV) that were used |

Output: `video` (H.264 MP4 preview), `seed`, and `metadata` (effective settings
and frame timing). `return_frames` adds a lossless PNG ZIP; `return_masks` adds
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

The worker limits prediction to 14 minutes and sampling to 12 minutes.
Requests whose estimated sampling time exceeds that budget are refused before
sampling. Platform startup needs a separate API deadline (see Runtime limits).
Historical measurements on an RTX PRO 6000, before the shorter deadline:

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
- Additional automatic masking backends; explicit multi-reference palette masks are supported now.

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

`cog build` downloads all eight inference weight files into the image, alongside the
BiRefNet cache. Startup validates their sizes and fails clearly if any file is
missing or truncated; it does not download replacement weights. The optional BF16 VAE adds 253.8 MB to the previous image sizes
reported below. The FP16 SCAIL checkpoint is not bundled. Replicate still needs to pull the image on a
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

The previously deployed FP8-only version includes seven weight files and is
39.12 GB uncompressed, compared with
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

## Sprute local / remote boundary

Sprute prepares the RGB 3×3 reference and driving grids plus both SCAIL palette
masks locally. This API only performs SCAIL inference; Sprute then splits the
returned frames, runs Toonout, and writes the transparent animation locally.

For the current `sprute-animate-character.api.json` sampling configuration:

```python
input = {
    "image": reference_grid_png,
    "video": driving_grid_mkv,
    "image_mask": reference_palette_png,
    "video_mask": driving_palette_mkv,
    "prepared_inputs": True,
    "preset": "balanced",             # UniPC, simple, 8 steps, CFG 1, shift 5
    "vae_precision": "bf16",
    "dpo_lora": 1.0,
    "lightx2v_lora": 0.8,             # applied after DPO, matching Sprute
    "mode": "animation",
    "seed": 42,
    "return_frames": True,
}
```

Prepared mode preserves the incoming canvas size (e.g. 576×768). Composite the
reference and driver over #808080 locally. Provide opaque RGB inputs; transparent
inputs are rejected in this mode. Masks must be exact 0/255 RGB palette colors,
not lossy H.264 masks. Driving masks must have matching size, frame count and FPS.
Automatic masks are disabled in prepared mode. Omitted explicit masks remain
absent from the graph. The existing 4n+1 frame rule and 161-frame cap still apply.
Use one request per motion. For exact timing, use FFV1 MKV; animated lossless WebP
is also accepted when frame durations are constant (within 1 ms rounding).

| Control | API input |
|---|---|
| Checkpoint / VAE | Bundled FP8 scaled checkpoint; `vae_precision` |
| Sampler / schedule | `sampler_name` (`preset`, `euler`, `uni_pc`), `scheduler` |
| Steps / CFG / shift | `steps`, `guidance_scale`, `shift` (0 inherits preset) |
| Distillation / DPO / relighting | `lightx2v_lora` (-1 inherits preset; 0 off), `dpo_lora`, `relight_lora` |
| Conditioning | `pose_strength`, `pose_start`, `pose_end`, `mode` |
| Denoising / reproducibility | `denoise`, `seed` |
| Dimensions / duration | `width`, `height`, `num_frames`, `fps` |
| Prepared masks | `image_mask`, `video_mask`, `prepared_inputs` |
| Text conditioning | `prompt`, `negative_prompt` |

`output.frames` is a ZIP of the original decoded PNGs, named `000000.png`,
`000001.png`, etc. Feed these to local Toonout rather than decoding the lossy MP4
preview. `output.metadata` records the effective FPS, dimensions, frame count,
seed, checkpoint and sampling settings. `return_masks` now returns the driving
mask as lossless FFV1 MKV to preserve palette colors.

The BF16 VAE is bundled during build alongside the default VAE.
The diffusion checkpoint is fixed to FP8 scaled. The previously deployed `b7d7a4fb`
version does **not** expose these new controls. Automatic continuation defaults
to five-frame overlap (`previous_frame_count`).
This API is not an arbitrary Comfy graph executor or a custom-LoRA loader.

### Additional views and continuation

`additional_images` accepts up to seven extra views, paired with
`additional_image_masks`. Also supply the primary `image_mask`. Each mask uses
the same identity colors as the primary view; their order must match the images.
The primary image supplies CLIP vision features, while all views supply VAE
reference conditioning.

`previous_frames` accepts a single image or a lossless video. The last
`previous_frame_count` frames anchor the beginning of the new request. Use count
1 for an image; otherwise the count must be 4n+1, up to 77 (trained default 5).
The driving clip must start at the overlapping interval, and the output includes
that interval. Drop it locally when appending to an earlier result. The count
also controls overlap between internally generated windows. Inputs shorter than
the requested anchor count are rejected rather than silently misaligned.

One prediction generates one sample. Use separate seeded requests for batches;
SAM3 tracking, skeleton extraction, custom LoRA downloads, sprite cutting and
background removal remain outside this GPU inference API.

### Current Sprute-compatible version

Pin `54cec44a0b05a60cde948cfa45ec4c61e0efd447ba85204796d40b8fa2112385`
for the controls documented above. Hosted prediction
`f4w7zg92knrny0d0xz0vt47npm` succeeded with prepared 576x768, 81-frame input
and returned 81 lossless PNGs. Inference took 74.89 seconds; total including
cold start was 540.42 seconds. See the validation document for settings and
limits of this test.

## Quantization

This deployment uses FP8-scaled quantized diffusion weights to reduce GPU memory usage. Outputs may differ from the original full-precision weights. The VAE defaults to BF16 to match the ComfyUI template.

## Output encoding

`output_format` selects `mp4` (H.264, default), `webm` (VP9), or looping animated `webp`. `output_quality` ranges from 1 to 100 (default 80); it controls compression, not inference. Values are codec-relative and 100 does not guarantee lossless RGB. Use `return_frames` for original PNGs. Outputs remain opaque; choosing WebP does not remove the background. Format and quality are recorded in metadata.
