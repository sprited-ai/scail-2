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
- **Quality or fast** — the official sampling recipe (UniPC, 40 steps, CFG 5)
  or the official ComfyUI recipe with the lightx2v step/CFG-distill LoRA
  (6 steps, CFG 1), about 10x cheaper. Steps / CFG / shift are exposed.
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
| `negative_prompt` | `""` | e.g. `distorted limbs, camera movement`; empty = Wan 2.1's standard negative prompt when CFG > 1 |
| `mode` | `animation` | `animation` or `replacement` |
| `image_mask` / `video_mask` | — | optional masks (see below) |
| `auto_mask` | `true` | derive missing masks (alpha channel → BiRefNet). Off = no masks |
| `resolution` | `512p` | short side 512 or 704; aspect follows the driving video (16:9 → 896x512, square → 512x512) |
| `width` / `height` | `0` | explicit size (multiples of 32) — centre-crops the driving video to that aspect |
| `num_frames` | `0` | 0 = all driving frames (max 161); rounded to 4k+1 |
| `fps` | `0` | resample the driving video to this rate first (nearest frame, no interpolation); 0 = source rate |
| `preset` | `quality` | `quality` = UniPC 40 steps / CFG 5 / shift 3; `fast` = lightx2v LoRA 0.8, Euler 6 steps / CFG 1 / shift 5 |
| `steps`, `guidance_scale`, `shift` | `0` | override the preset (0 = preset default) |
| `dpo_lora` | `0` | strength of the official Bias-Aware DPO LoRA (the ComfyUI template uses 1.0 with `fast`) |
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
