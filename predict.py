"""SCAIL-2 (zai-org) end-to-end character animation on Replicate. (build 2026-09-18a)

Runs the official ComfyUI implementation (WanSCAILToVideo, ComfyUI v0.33.1)
headless, on the Comfy-Org fp8 repack of the 14B model. Inputs are prepared
here (sizing, fps resampling, SCAIL mask colouring, optional BiRefNet
auto-masks), the graph is built by workflow.py, and ComfyUI does the sampling.
See README.md for the input semantics.
"""
from __future__ import annotations

import glob
import json
import os
import random
import shutil
import subprocess
import time
import uuid
from fractions import Fraction
from typing import Optional

import av
import numpy as np
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image

import preprocess as pp
from comfy_client import ComfyServer
from matting import BiRefNetMatter
from workflow import LORA_DPO, LORA_RELIGHT, LORA_LIGHTX2V, UNET, VAE, VAE_BF16, PRESETS, GraphParams, build_graph
from runtime_limits import deadline

COMFY_DIR = os.environ.get("COMFY_DIR", "/ComfyUI")
WORK = os.environ.get("SCAIL2_WORK", "/tmp/scail2")
INPUT_DIR, OUTPUT_DIR, TEMP_DIR = f"{WORK}/input", f"{WORK}/output", f"{WORK}/temp"

HF = "https://huggingface.co"
# path under COMFY_DIR/models -> (build-time download URL, size in bytes).
# cog.yaml bundles these files; setup() only verifies them.
WEIGHTS = {
    "vae/Wan2_1_VAE_bf16.safetensors":
        (f"{HF}/Kijai/WanVideo_comfy/resolve/8260d429d19fd7a72304cad059160b95d843913f/Wan2_1_VAE_bf16.safetensors", 253806278),
    "diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors":
        (f"{HF}/Comfy-Org/SCAIL-2/resolve/main/diffusion_models/wan2.1_14B_SCAIL_2_fp8_scaled.safetensors", 17694586857),
    "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors":
        (f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", 6735906897),
    "vae/wan_2.1_vae.safetensors":
        (f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors", 253815318),
    "clip_vision/clip_vision_h.safetensors":
        (f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors", 1264219396),
    "loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors":
        (f"{HF}/Comfy-Org/SCAIL-2/resolve/main/loras/wan2.1_SCAIL_2_DPO_lora_bf16.safetensors", 1226936552),
    "loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors":
        (f"{HF}/Comfy-Org/SCAIL-2/resolve/main/loras/wan2.1_SCAIL_2_relight_lora_bf16.safetensors", 1226936552),
    "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors":
        (f"{HF}/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", 738005744),
}
REQUIRED_NODES = ("WanSCAILToVideo", "LoadVideo", "GetVideoComponents", "ImageFromBatch", "ImageBatch", "SaveImage")
MAX_PROBE_FRAMES = 20000  # refuse to scan driving videos longer than this
# Leave headroom for preprocessing/encoding within the hard predict deadline.
MAX_SAMPLING_SECONDS = 12 * 60
EFFECTIVE_TFLOPS = float(os.environ.get("SCAIL2_TFLOPS", 350))


def ensure_weights() -> None:
    """Validate image-bundled weights. Runtime startup never downloads models."""
    for rel, (_, size) in WEIGHTS.items():
        dest = os.path.join(COMFY_DIR, "models", rel)
        if not os.path.isfile(dest):
            raise RuntimeError(f"Missing bundled weight: {dest}. Rebuild the Cog image.")
        got = os.path.getsize(dest)
        if got != size:
            raise RuntimeError(f"{rel}: expected {size} bytes, got {got}. Rebuild the Cog image.")


# ----------------------------------------------------------------------------
# video helpers (PyAV)
# ----------------------------------------------------------------------------

def is_webp(path: str) -> bool:
    with open(path, "rb") as handle:
        header = handle.read(12)
    return header[:4] == b"RIFF" and header[8:12] == b"WEBP"


def probe_video(path: str) -> tuple[int, float, int, int]:
    """(frame count, fps, width, height). Counts frames by decoding when the
    container does not say (webm/mkv often don't)."""
    if is_webp(path):
        with Image.open(path) as im:
            count = getattr(im, "n_frames", 1)
            durations = []
            for index in range(count):
                im.seek(index)
                im.load()
                durations.append(im.info.get("duration", 0))
            if count < 2 or not all(durations) or max(durations) - min(durations) > 1:
                raise ValueError("Driving WebP must be animated with a constant frame duration")
            return count, 1000 * count / sum(durations), im.width, im.height
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or s.base_rate or 24)
        w, h = s.codec_context.width, s.codec_context.height
        n = int(s.frames or 0)
        if n <= 0:
            n = 0
            for _ in c.decode(s):
                n += 1
                if n > MAX_PROBE_FRAMES:
                    raise ValueError(f"driving video has more than {MAX_PROBE_FRAMES} frames")
    return n, fps, w, h


def decode_selected(path: str, indices: list[int], convert) -> list:
    """Decode frames whose index is in `indices` (sorted, may repeat) and map
    each through `convert(PIL.Image) -> object`. Stops after the last one."""
    if is_webp(path):
        with Image.open(path) as im:
            result = []
            for index in indices:
                im.seek(index)
                result.append(convert(im.copy()))
            return result
    wanted: dict[int, int] = {}
    for i in indices:
        wanted[i] = wanted.get(i, 0) + 1
    last = max(indices)
    by_index: dict[int, object] = {}
    with av.open(path) as c:
        for i, frame in enumerate(c.decode(video=0)):
            if i in wanted:
                by_index[i] = convert(frame.to_image())
            if i >= last:
                break
    missing = [i for i in indices if i not in by_index]
    if missing:  # container over-reported its frame count: hold the last decoded frame
        tail = by_index[max(by_index)]
        for i in missing:
            by_index[i] = tail
    return [by_index[i] for i in indices]


def write_video(path: str, frames: list[np.ndarray], fps: float, lossless: bool = True) -> None:
    """Write HxWx3 uint8 frames. Lossless FFV1/MKV for ComfyUI inputs (mask
    colours must survive exactly); H.264/MP4 otherwise."""
    rate = Fraction(fps).limit_denominator(1000)
    h, w = frames[0].shape[:2]
    with av.open(path, "w") as c:
        if lossless:
            s = c.add_stream("ffv1", rate=rate)
            s.pix_fmt = "bgr0"
        else:
            s = c.add_stream("libx264", rate=rate)
            s.pix_fmt = "yuv420p"
            s.options = {"crf": "12", "preset": "medium"}
        s.width, s.height = w, h
        for f in frames:
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24").reformat(format=s.pix_fmt)
            for pkt in s.encode(vf):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def encode_mp4(frame_glob_pattern: str, n_frames: int, fps: float, out: str, crf: int = 15) -> None:
    subprocess.check_call([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-framerate", f"{fps:.6f}", "-start_number", "1",
        "-i", frame_glob_pattern, "-frames:v", str(n_frames),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "medium",
        "-movflags", "+faststart", out,
    ], stdin=subprocess.DEVNULL)


# ----------------------------------------------------------------------------

class Output(BaseModel):
    video: Path
    seed: int
    frames: Optional[Path] = None
    metadata: Optional[Path] = None
    reference_mask: Optional[Path] = None
    driving_mask: Optional[Path] = None


class Predictor(BasePredictor):
    def setup(self) -> None:
        with deadline(600, "setup"):
            t0 = time.time()
            print("[setup] validating bundled weights", flush=True)
            ensure_weights()
            print("[setup] bundled weights verified; starting ComfyUI", flush=True)
            for d in (INPUT_DIR, OUTPUT_DIR, TEMP_DIR):
                os.makedirs(d, exist_ok=True)
            extra = tuple(os.environ.get("COMFY_EXTRA_ARGS", "").split())  # e.g. "--cpu" for CPU smoke tests
            self.comfy = ComfyServer(COMFY_DIR, INPUT_DIR, OUTPUT_DIR, TEMP_DIR, extra_args=extra)
            self.comfy.start()
            self.comfy.wait_ready()
            info = self.comfy.object_info()
            missing = [n for n in REQUIRED_NODES if n not in info]
            if missing:
                raise RuntimeError(f"ComfyUI is missing nodes {missing}; is the checkout the pinned version?")
            self.matter = BiRefNetMatter("cuda")
            print(f"[setup] ready in {time.time() - t0:.0f}s", flush=True)

    def predict(
        self,
        image: Path = Input(description="Reference character image. Transparent PNGs are composited on white and their alpha becomes the reference mask."),
        video: Path = Input(description="Driving video (mp4/mov/webm/mkv or constant-rate animated WebP). Its motion is transferred to the character; its aspect ratio sets the output size unless width/height are given."),
        prompt: str = Input(default="", description="Describe the character and the motion, e.g. 'A cartoon robot walking in place, side view'. Describes the final video; not instructions."),
        negative_prompt: str = Input(default="", description="What to avoid, e.g. 'distorted limbs, camera movement, blurry'. Only matters with guidance_scale > 1 (the quality preset). Wan's stock Chinese negative prompt is deliberately not applied: its 'painting / artwork / style' terms push stylized characters toward a 3D-CG look."),
        mode: str = Input(default="animation", choices=["animation", "replacement"],
                          description="animation: the reference character (and its background) performs the driving motion. replacement: the character is placed into the driving video, keeping its background and lighting."),
        image_mask: Optional[Path] = Input(default=None, description="Optional mask for the reference: a grayscale/black-and-white matte (white = character) or a SCAIL-2 palette mask (blue = identity 0). If omitted and auto_mask is on, one is derived from the image's alpha channel or BiRefNet."),
        video_mask: Optional[Path] = Input(default=None, description="Optional per-frame mask video for the driving video, same conventions as image_mask (grayscale matte or SCAIL-2 colours). If omitted and auto_mask is on, BiRefNet masks every frame."),
        auto_mask: bool = Input(default=True, description="Derive missing masks automatically (alpha channel or BiRefNet single-subject matting). Off = run without masks."),
        resolution: str = Input(default="512p", choices=["512p", "704p"], description="Short side of the output; the driving video's aspect ratio is kept (e.g. 896x512 for 16:9, 512x512 for square). SCAIL-2 was trained at both."),
        width: int = Input(default=0, ge=0, le=1536, description="Explicit output width (multiple of 32). Set together with height to override resolution; the driving video is centre-cropped to this aspect."),
        height: int = Input(default=0, ge=0, le=1536, description="Explicit output height (multiple of 32)."),
        num_frames: int = Input(default=0, ge=0, le=pp.MAX_FRAMES, description=f"Frames to generate (rounded down to 4k+1). 0 = every frame of the driving video, up to {pp.MAX_FRAMES}. Beyond 81 frames the video is generated in overlapping windows."),
        fps: int = Input(default=0, ge=0, le=60, description="Resample the driving video to this frame rate before animating (frames are held, never interpolated); the output uses the same rate. 0 = keep the driving video's rate."),
        preset: str = Input(default="fast", choices=["fast", "quality", "sprute"],
                            description="fast (default): the official ComfyUI recipe — lightx2v step/CFG-distill LoRA, Euler, 6 steps, CFG 1, shift 5; sprute: UniPC, 8 steps, CFG 1, shift 5, DPO before LightX2V. Model/VAE precision are selected separately. quality: the paper's sampler (UniPC, 40 steps, CFG 5, shift 3) — 81 frames at 512p takes ~10 min on an H100."),
        prepared_inputs: bool = Input(default=False, description="Already-composited RGB grids and exact RGB SCAIL palette masks from local Sprute. No resize, crop, recoloring or automatic masks. Inputs must have matching dimensions; video masks must match frame count and FPS. Use lossless WebP or FFV1 MKV."),
        additional_images: Optional[list[Path]] = Input(default=None, description="Up to 7 additional reference views. Supply a matching additional_image_masks list and image_mask; palette colors bind views to the same identities. CLIP vision uses the primary image."),
        additional_image_masks: Optional[list[Path]] = Input(default=None, description="Palette mask for each additional reference, in the same order."),
        previous_frames: Optional[Path] = Input(default=None, description="Previous output as an image or lossless video. Its tail anchors the beginning of this request. The driving video must include that overlapping interval, and returned frames include the anchor interval."),
        previous_frame_count: int = Input(default=5, ge=1, le=77, description="Tail frames used for anchoring and chunk overlap; must be 4n+1. Use 1 for a single-image anchor. SCAIL-2 was trained with 5."),
        vae_precision: str = Input(default="default", choices=["default", "bf16"], description="Select VAE checkpoint; Sprute uses bf16."),
        sampler_name: str = Input(default="preset", choices=["preset", "euler", "uni_pc"], description="Override preset sampler."),
        scheduler: str = Input(default="simple", choices=["simple", "normal", "karras", "exponential", "sgm_uniform", "ddim_uniform", "beta", "linear_quadratic", "kl_optimal"], description="ComfyUI sampling schedule."),
        lightx2v_lora: float = Input(default=-1, ge=-1, le=2, description="LightX2V strength: -1 uses preset, 0 disables. Sprute uses 0.8."),
        pose_start: float = Input(default=0, ge=0, le=1, description="Start fraction of pose conditioning."),
        pose_end: float = Input(default=1, ge=0, le=1, description="End fraction of pose conditioning."),
        denoise: float = Input(default=1, ge=0.001, le=1, description="KSampler denoise strength."),
        return_frames: bool = Input(default=False, description="Return original decoded PNG frames as ZIP for lossless local Toonout and sprite assembly. MP4 is only a preview."),
        steps: int = Input(default=0, ge=0, le=100, description="Sampling steps; 0 = preset default."),
        guidance_scale: float = Input(default=0, ge=0, le=20, description="Classifier-free guidance; 0 = preset default (5 quality / 1 fast)."),
        shift: float = Input(default=0, ge=0, le=20, description="Flow-matching schedule shift; 0 = preset default (3 quality / 5 fast)."),
        dpo_lora: float = Input(default=1.0, ge=0, le=2, description="Strength of the official Bias-Aware DPO LoRA (0 = off). 1.0 is what the official ComfyUI template ships with; the released base checkpoint is pre-DPO."),
        relight_lora: float = Input(default=0, ge=0, le=2, description="Strength of the official relighting LoRA for replacement mode (0 = off). Improves lighting consistency with the driving scene."),
        pose_strength: float = Input(default=1.0, ge=0, le=10, description="Weight of the driving-motion conditioning."),
        seed: Optional[int] = Input(default=None, description="Random seed; leave empty for random."),
        return_masks: bool = Input(default=False, description="Also return the reference/driving masks that were used (for debugging or re-use as image_mask / video_mask)."),
    ) -> Output:
        with deadline(840, "predict"):
            t0 = time.time()
            if not self.comfy.alive():
                self.comfy.start()
                self.comfy.wait_ready()
            job = uuid.uuid4().hex[:12]
            if seed is None:
                seed = random.randrange(2**31)
            if pose_start > pose_end:
                raise ValueError("pose_start must be <= pose_end")
            additional_images = additional_images or []
            additional_image_masks = additional_image_masks or []
            if len(additional_images) > 7 or len(additional_images) != len(additional_image_masks):
                raise ValueError("Provide up to 7 additional images with one matching mask each")
            if additional_images and image_mask is None:
                raise ValueError("Multi-reference conditioning requires image_mask for the primary reference")
            replacement = mode == "replacement"
            drive_bg = "white" if replacement else "black"   # driving mask background per SCAIL-2 convention
            ref_bg = "black" if replacement else "white"     # reference mask background (the opposite)
            if relight_lora > 0 and not replacement:
                print("[warn] relight_lora is meant for replacement mode; applying anyway", flush=True)

            # ---- driving video -> frames at the output size ---------------------
            n_src, src_fps, sw, sh = probe_video(str(video))
            if n_src == 0:
                raise ValueError("could not read any frames from the driving video")
            W, H = pp.target_size(sw, sh, resolution, width, height)
            if prepared_inputs:
                W, H = pp.target_size(sw, sh, resolution, width or sw, height or sh)
                if (W, H) != (sw, sh):
                    raise ValueError("Prepared driving video must already match output dimensions")
                auto_mask = False
            limit = min(num_frames or pp.MAX_FRAMES, pp.MAX_FRAMES)
            sel = pp.select_frames(n_src, src_fps, fps, limit)
            if len(sel) < 5:
                raise ValueError(f"driving video too short: {len(sel)} usable frames (need at least 5)")
            n = len(sel)
            out_fps = float(fps) if fps > 0 else src_fps
            plan = pp.plan_chunks(n, overlap=previous_frame_count)
            print(f"[input] driver {sw}x{sh}@{src_fps:.3f}fps x{n_src} -> {W}x{H}@{out_fps:.3f}fps x{n} "
                  f"(windows: {len(plan.starts)} x {plan.length})", flush=True)
            frames = decode_selected(str(video), sel, lambda im: pp.prepared_rgb(im, W, H) if prepared_inputs else np.asarray(pp.cover(im.convert("RGB"), W, H)))

            # ---- driving mask ---------------------------------------------------
            drive_mask: list[np.ndarray] | None = None
            if video_mask is not None:
                m_n, m_fps, m_w, m_h = probe_video(str(video_mask))
                if prepared_inputs and (m_n != n_src or (m_w, m_h) != (W, H) or abs(m_fps - src_fps) > 0.01):
                    raise ValueError("Prepared video_mask must match driving dimensions, frame count and FPS")
                m_sel = pp.map_indices(sel, n_src, m_n)
                drive_mask = decode_selected(str(video_mask), m_sel, lambda im: pp.prepared_rgb(im, W, H, palette=True) if prepared_inputs else pp.mask_from_image(im, drive_bg, W, H, "cover"))
            elif auto_mask:
                t = time.time()
                drive_mask = [pp.colorize(m, drive_bg) for m in self.matter.masks(frames)]
                print(f"[mask] BiRefNet on {n} driving frames in {time.time() - t:.1f}s", flush=True)

            # ---- reference image + mask ----------------------------------------
            ref_rgb, alpha = pp.flatten_alpha(Image.open(str(image)))
            ref = Image.fromarray(pp.prepared_rgb(Image.open(str(image)), W, H)) if prepared_inputs else pp.letterbox(ref_rgb, W, H, pp.WHITE)
            ref_mask: np.ndarray | None = None
            if image_mask is not None:
                ref_mask = pp.prepared_rgb(Image.open(str(image_mask)), W, H, palette=True) if prepared_inputs else pp.mask_from_image(Image.open(str(image_mask)), ref_bg, W, H, "letterbox")
            elif auto_mask:
                if alpha is not None and alpha.min() < 128:
                    ref_mask = pp.mask_from_alpha(alpha, ref_bg, W, H)
                    print("[mask] reference mask from alpha channel", flush=True)
                else:
                    ref_mask = pp.colorize(self.matter.masks([np.asarray(ref)])[0], ref_bg)
                    print("[mask] reference mask from BiRefNet", flush=True)

            extra_refs = []
            for extra, mask in zip(additional_images, additional_image_masks):
                with Image.open(str(extra)) as im:
                    rgb = pp.prepared_rgb(im, W, H) if prepared_inputs else np.asarray(pp.letterbox(pp.flatten_alpha(im)[0], W, H, pp.WHITE))
                with Image.open(str(mask)) as im:
                    matte = pp.prepared_rgb(im, W, H, palette=True) if prepared_inputs else pp.mask_from_image(im, ref_bg, W, H, "letterbox")
                extra_refs.append((rgb, matte))
            anchors = None
            if previous_frames is not None:
                try:
                    anchor_image = Image.open(str(previous_frames))
                except OSError:
                    anchor_image = None  # Not a Pillow image: decode as video below.
                if anchor_image is not None:
                    with anchor_image as im:
                        if getattr(im, "n_frames", 1) == 1:
                            anchors = [pp.prepared_rgb(im, W, H) if prepared_inputs else np.asarray(pp.cover(im.convert("RGB"), W, H))]
                if anchors is None:
                    count, _, _, _ = probe_video(str(previous_frames))
                    anchors = decode_selected(str(previous_frames), list(range(max(0, count - previous_frame_count), count)),
                                              lambda im: pp.prepared_rgb(im, W, H) if prepared_inputs else np.asarray(pp.cover(im.convert("RGB"), W, H)))
                if len(anchors) < previous_frame_count or previous_frame_count >= n:
                    raise ValueError("Previous frames must supply previous_frame_count frames, fewer than num_frames; use count=1 for an image")

            # ---- stage inputs for ComfyUI ---------------------------------------
            pad = plan.total - n
            ref_name, drive_name = f"{job}-ref.png", f"{job}-drive.mkv"
            ref.save(os.path.join(INPUT_DIR, ref_name))
            write_video(os.path.join(INPUT_DIR, drive_name), frames + [frames[-1]] * pad, out_fps)
            ref_mask_name = drive_mask_name = None
            if ref_mask is not None:
                ref_mask_name = f"{job}-refmask.png"
                Image.fromarray(ref_mask).save(os.path.join(INPUT_DIR, ref_mask_name))
            if drive_mask is not None:
                drive_mask_name = f"{job}-drivemask.mkv"
                write_video(os.path.join(INPUT_DIR, drive_mask_name), drive_mask + [drive_mask[-1]] * pad, out_fps)
            extra_names = []
            for i, (rgb, matte) in enumerate(extra_refs):
                names = (f"{job}-extra{i}.png", f"{job}-extra{i}-mask.png")
                Image.fromarray(rgb).save(os.path.join(INPUT_DIR, names[0]))
                Image.fromarray(matte).save(os.path.join(INPUT_DIR, names[1]))
                extra_names.append(names)
            previous_name = f"{job}-previous.mkv" if anchors is not None else None
            if previous_name:
                write_video(os.path.join(INPUT_DIR, previous_name), anchors, out_fps)

            # ---- sampling settings ---------------------------------------------
            pr = PRESETS[preset]
            cfg = guidance_scale if guidance_scale > 0 else pr["cfg"]
            strength = (pr["loras"][0][1] if pr["loras"] else 0) if lightx2v_lora < 0 else lightx2v_lora
            loras = []
            if preset != "sprute" and strength > 0:
                loras.append((LORA_LIGHTX2V, strength))
            if dpo_lora > 0:
                loras.append((LORA_DPO, dpo_lora))
            if preset == "sprute" and strength > 0:
                loras.append((LORA_LIGHTX2V, strength))
            if relight_lora > 0:
                loras.append((LORA_RELIGHT, relight_lora))
            params = GraphParams(
                ref_image=ref_name, drive_video=drive_name, ref_mask=ref_mask_name, drive_mask=drive_mask_name,
                prompt=prompt, negative_prompt=negative_prompt, width=W, height=H, plan=plan, seed=seed,
                steps=steps or pr["steps"], cfg=cfg,
                shift=shift if shift > 0 else pr["shift"], sampler=pr["sampler"] if sampler_name == "preset" else sampler_name, scheduler=scheduler,
                unet=UNET,
                vae=VAE_BF16 if vae_precision == "bf16" else VAE,
                pose_start=pose_start, pose_end=pose_end, denoise=denoise,
                additional_refs=extra_names, previous_frames=previous_name,
                output_prefix=f"{job}/f", replacement=replacement, pose_strength=pose_strength, loras=loras,
            )
            est = pp.estimate_seconds(W, H, plan, params.steps, params.cfg, EFFECTIVE_TFLOPS)
            print(f"[sample] preset={preset} steps={params.steps} cfg={params.cfg} shift={params.shift} "
                  f"sampler={params.sampler} loras={loras} seed={seed} mode={mode} est~{est / 60:.1f}min", flush=True)
            if est > MAX_SAMPLING_SECONDS:
                raise ValueError(
                    f"this request would sample for ~{est / 60:.0f} min ({len(plan.starts)} window(s) of {plan.length} frames at {W}x{H}, "
                    f"{params.steps} steps, CFG {params.cfg}), exceeding our 12-minute sampling budget. "
                    "Use preset='fast', fewer num_frames, or a smaller resolution.")

            # ---- run -------------------------------------------------------------
            out_dir = os.path.join(OUTPUT_DIR, job)
            try:
                t = time.time()
                self.comfy.run(build_graph(params), timeout=MAX_SAMPLING_SECONDS)
                print(f"[sample] done in {time.time() - t:.0f}s", flush=True)
                produced = sorted(glob.glob(os.path.join(out_dir, "f_*_.png")))
                if len(produced) < n:
                    raise RuntimeError(f"expected {n} frames, ComfyUI produced {len(produced)}")
                out_path = os.path.join(WORK, f"{job}.mp4")
                encode_mp4(os.path.join(out_dir, "f_%05d_.png"), n, out_fps, out_path)
                result = Output(video=Path(out_path), seed=seed)
                if return_frames:
                    archive = os.path.join(WORK, f"{job}-frames.zip")
                    pp.archive_frames(produced[:n], archive)
                    result.frames = Path(archive)
                metadata_path = os.path.join(WORK, f"{job}-metadata.json")
                with open(metadata_path, "w") as handle:
                    json.dump(dict(seed=seed, width=W, height=H, num_frames=n, fps=out_fps,
                                   model=params.unet, vae=params.vae, steps=params.steps, cfg=params.cfg,
                                   sampler=params.sampler, scheduler=params.scheduler, shift=params.shift,
                                   loras=loras, mode=mode, pose_strength=pose_strength,
                                   pose_start=pose_start, pose_end=pose_end, denoise=denoise,
                                   reference_count=1 + len(extra_names), previous_frame_count=previous_frame_count,
                                   previous_frames_supplied=previous_frames is not None,
                                   prepared_inputs=prepared_inputs), handle, indent=2)
                result.metadata = Path(metadata_path)
                if return_masks:
                    if ref_mask is not None:
                        rm = os.path.join(WORK, f"{job}-reference-mask.png")
                        Image.fromarray(ref_mask).save(rm)
                        result.reference_mask = Path(rm)
                    if drive_mask is not None:
                        dm = os.path.join(WORK, f"{job}-driving-mask.mkv")
                        write_video(dm, drive_mask, out_fps, lossless=True)
                        result.driving_mask = Path(dm)
            finally:
                for name in (ref_name, drive_name, ref_mask_name, drive_mask_name, previous_name, *(name for pair in extra_names for name in pair)):
                    if name:
                        try:
                            os.remove(os.path.join(INPUT_DIR, name))
                        except OSError:
                            pass
                shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[done] {n} frames {W}x{H} in {time.time() - t0:.0f}s", flush=True)
            return result
