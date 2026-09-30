# SCAIL-2

Animate a character image using the motion from a driving video, or replace a character in the driving scene.

This is an unofficial community deployment of [SCAIL-2 by Zhipu AI](https://github.com/zai-org/SCAIL-2).

## Examples

[View the examples](https://replicate.com/sprited/scail-2/examples), including a clay gardener and a robot. Examples use MP4 output without a PNG frame archive.

## Start here

1. Upload a reference character image and a driving video.
2. Choose **animation** to animate the reference character in its own scene, or **replacement** to preserve the driving scene.
3. Start with **fast**. Set a seed to repeat a configuration.

The result includes an MP4, WebM, or animated WebP and settings JSON. Enable **return_frames** for a ZIP of lossless PNG frames, or **return_masks** to inspect and reuse the masks.

## Presets

- **fast** (default): ComfyUI template sampling settings — Euler, 6 steps, CFG 1, shift 5, DPO 1.0 followed by LightX2V 0.8.
- **balanced**: UniPC, 8 steps, CFG 1, shift 5, with the same LoRA strengths.
- **quality**: UniPC, 40 steps, CFG 5, shift 3, without the distillation LoRA. This takes substantially longer and is not guaranteed to improve every input.

This deployment uses **FP8-scaled quantized diffusion model weights** to reduce GPU memory usage. Outputs may differ from the original full-precision weights. The **VAE uses BF16 by default**, matching the ComfyUI template. Sampling controls can be overridden individually.

## Masks and transparency

An explicit **video_mask** takes priority. Otherwise, an animated WebP's transparency is used as the driving mask. If neither is available, **auto_mask** generates a single-subject mask with BiRefNet. Transparent reference images also provide a reference mask when auto_mask is enabled.

You can supply grayscale masks (white = subject) or SCAIL-2 color masks. For exact preprocessed RGB images and palette masks, enable **prepared_inputs**.

## Video length

**num_frames** limits frames from the beginning of the driving video after applying **fps**. Zero uses the available input, up to 161 frames. At 24 fps, 81 frames is approximately 3.4 seconds. A larger number does not extend a shorter input. Frame counts are trimmed to supported lengths: 5, 9, 13, and so on.

**fps** defaults to the source frame rate. Changing it resamples the input by repeating or dropping frames; it does not interpolate motion.

## Additional controls

Additional reference images and matching masks can describe more views of the same character. Previous frames can anchor a continuation; the returned video includes the overlap, which callers should trim when joining clips.

Cold starts include loading the model image and can take several minutes. API callers can set `Cancel-After: 10m` to bound a request including startup. This header is per request; it is not a default timeout for Playground runs.

## Source

[Deployment source and technical details](https://github.com/sprited-ai/scail-2-on-replicate). See the upstream project and deployment repository for component licenses and model attribution.

## Output format and quality

Choose `output_format`: `mp4` (default), `webm`, or animated `webp`. `output_quality` sets compression quality from 1 to 100 (default 80). Higher values generally produce larger files; values are not comparable across codecs and do not measure inference quality. For a video without additional frame archives, set `return_frames` to false. WebP output does not automatically remove the background.
