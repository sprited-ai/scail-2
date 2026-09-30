# Sprute inference controls validation

Validated on gin on 2026-09-30 UTC. The GPU was shared with unrelated processes;
these timings are not isolated performance or FP8/FP16 comparisons.

## Requirements and evidence

- Current Sprute graph: `sprute/workflows/sprute-animate-character.api.json`.
  FP16 SCAIL-2, BF16 VAE, DPO 1 then LightX2V 0.8, UniPC/simple, 8 steps,
  CFG 1, shift 5, denoise 1, pose strength/start/end 1/0/1.
- Prepared RGB grids retain their exact pixels and dimensions. Palette masks
  survive a real FFV1 encode/decode. Tests exercise the predictor request path.
- PNG ZIP outputs retain original PNG bytes and frame order. Metadata records
  effective configuration and timing. MP4 remains a preview.
- Multi-reference images/masks and previous-frame anchors reach the official
  conditioning node. Adjustable overlap is consistent across window offsets,
  node anchor counts and output concatenation.
- `python -m pytest -q`: 35 tests passed.

## Container HTTP tests

Image: `scail2-controls:test`, approximately 72.2 GB uncompressed. Rebuilt with
Cog 0.23.0 and `cog.python-base.yaml --use-cuda-base-image=false`, including the
regenerated API schema and all model weights. No network or model mounts.
Fixture assets were mounted read-only. `COMFY_EXTRA_ARGS=--lowvram` was used
because this was a shared GPU. Each harness had a 580-second outer deadline
and stopped its container in `finally`.

1. Prepared 576x768 eight-direction grid, 5 driver frames at 24 FPS, seed 42,
   Sprute preset, FP16 model/BF16 VAE, both explicit masks, PNG frame output.
   Ready: 6.77 s. Prediction: 40.04 s. Succeeded; all five PNGs, metadata and
   preview retrieved. Output frame inspected.
2. Same settings plus an additional reference/mask and a single-image anchor
   (`previous_frame_count=1`). Ready: 6.79 s. Prediction: 47.74 s. Succeeded;
   metadata confirms both controls, five PNGs returned and output inspected.

Remote evidence: `/mnt/stash/scail2-controls/` (`response.json`, `container.log`,
`result.zip`, `result.json`) and `extended/`. Local inspected samples:
`output/sprute-controls/frame.png` and `extended.png` (not committed).

These short tests verify execution and transport. They do not establish
full-length animation quality, same-seed equality across hardware, multi-view
quality improvement, or hosted execution of this new version. The older public
FP8 version's hosted success is documented in README.

## FP8 deployment selection

The deployment now bundles only the FP8 scaled SCAIL checkpoint, by user choice.
The FP16 results above describe the earlier comparison image, not the current
public API. VAE precision remains configurable. SageAttention is not yet enabled
in the deployment; the experimental gin package lacks an SM90 binary.

Same prepared input, seed 42, 576x768, 81 frames, Sprute preset, BF16 VAE,
lowvram, sequential runs on gin (RTX PRO 6000 Blackwell):

| Model | Attention | Prediction seconds | Process peak GPU MiB |
|---|---|---:|---:|
| FP16 | PyTorch SDPA | 207.42 | 46238 |
| FP8 scaled | PyTorch SDPA | 144.46 | 33278 |
| FP16 | SageAttention 2.2.0 | 167.80 | 46238 |
| FP8 scaled | SageAttention 2.2.0 | 106.93 | 33278 |

One measured run per configuration, not a statistical benchmark or H100 result.
Memory is sampled once per second for container process IDs with nvidia-smi;
PyTorch allocator peaks alone omit dynamic weight allocations. Evidence lives
in `/mnt/stash/scail2-controls/bench-{fp16,fp8}-{sdpa,sage}/`, including
response.json, container.log, result.zip and gpu-memory.json.

The dual-checkpoint upload terminated with HTTP 413 from the registry gateway.
FP8-only packaging is being uploaded separately; hosted verification is pending.
