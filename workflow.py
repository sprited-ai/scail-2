"""Build the ComfyUI API-format graph for one SCAIL-2 prediction.

Pure function of its parameters (no I/O) so it can be unit-tested against the
node schemas dumped from a live ComfyUI (tests/test_workflow.py). Node ids are
strings; links are ["node_id", output_index].
"""
from __future__ import annotations

from dataclasses import dataclass, field

from preprocess import OVERLAP, ChunkPlan

UNET = "wan2.1_14B_SCAIL_2_fp8_scaled.safetensors"
TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"
LORA_LIGHTX2V = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
LORA_DPO = "wan2.1_SCAIL_2_DPO_lora_bf16.safetensors"
LORA_RELIGHT = "wan2.1_SCAIL_2_relight_lora_bf16.safetensors"

# Sampling presets. "quality" = the official generate.py defaults (unipc, 40
# steps, cfg 5, shift 3). "fast" = the official ComfyUI template: lightx2v
# step/cfg-distill LoRA at 0.8, euler, 6 steps, cfg 1, shift 5.
PRESETS = {
    "quality": dict(steps=40, cfg=5.0, shift=3.0, sampler="uni_pc", scheduler="simple", loras=[]),
    "fast": dict(steps=6, cfg=1.0, shift=5.0, sampler="euler", scheduler="simple", loras=[(LORA_LIGHTX2V, 0.8)]),
}


@dataclass
class GraphParams:
    ref_image: str                 # filename inside ComfyUI's input dir
    drive_video: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    plan: ChunkPlan
    seed: int
    steps: int
    cfg: float
    shift: float
    sampler: str
    scheduler: str
    output_prefix: str             # SaveImage filename_prefix (may contain a subfolder)
    ref_mask: str | None = None
    drive_mask: str | None = None
    replacement: bool = False
    pose_strength: float = 1.0
    loras: list[tuple[str, float]] = field(default_factory=list)


def build_graph(p: GraphParams) -> dict:
    g: dict[str, dict] = {}

    def node(nid: str, class_type: str, **inputs):
        g[nid] = {"class_type": class_type, "inputs": inputs}
        return [nid, 0]

    model = node("unet", "UNETLoader", unet_name=UNET, weight_dtype="default")
    for i, (name, strength) in enumerate(p.loras):
        model = node(f"lora{i}", "LoraLoaderModelOnly", model=model, lora_name=name, strength_model=strength)
    model = node("shift", "ModelSamplingSD3", model=model, shift=p.shift)

    clip = node("clip", "CLIPLoader", clip_name=TEXT_ENCODER, type="wan")
    vae = node("vae", "VAELoader", vae_name=VAE)
    clip_vision = node("clip_vision", "CLIPVisionLoader", clip_name=CLIP_VISION)

    ref = node("ref", "LoadImage", image=p.ref_image)
    cv_out = node("clip_vision_encode", "CLIPVisionEncode", clip_vision=clip_vision, image=ref, crop="none")
    ref_mask = node("ref_mask", "LoadImage", image=p.ref_mask) if p.ref_mask else None

    drive = node("drive", "LoadVideo", file=p.drive_video)
    drive_frames = node("drive_frames", "GetVideoComponents", video=drive)
    drive_mask_frames = None
    if p.drive_mask:
        drive_mask = node("drive_mask", "LoadVideo", file=p.drive_mask)
        drive_mask_frames = node("drive_mask_frames", "GetVideoComponents", video=drive_mask)

    positive = node("positive", "CLIPTextEncode", clip=clip, text=p.prompt)
    negative = node("negative", "CLIPTextEncode", clip=clip, text=p.negative_prompt)

    previous = None
    acc = None
    for i, offset in enumerate(p.plan.offsets):
        cond = dict(
            positive=positive, negative=negative, vae=vae,
            width=p.width, height=p.height, length=p.plan.length, batch_size=1,
            pose_strength=p.pose_strength, pose_start=0.0, pose_end=1.0,
            video_frame_offset=offset, previous_frame_count=OVERLAP,
            replacement_mode=p.replacement,
            reference_image=ref, clip_vision_output=cv_out, pose_video=drive_frames,
        )
        if ref_mask is not None:
            cond["reference_image_mask"] = ref_mask
        if drive_mask_frames is not None:
            cond["pose_video_mask"] = drive_mask_frames
        if previous is not None:
            cond["previous_frames"] = previous
        node(f"scail{i}", "WanSCAILToVideo", **cond)
        latent = node(f"sample{i}", "KSampler", model=model, seed=p.seed, steps=p.steps, cfg=p.cfg,
                      sampler_name=p.sampler, scheduler=p.scheduler,
                      positive=[f"scail{i}", 0], negative=[f"scail{i}", 1], latent_image=[f"scail{i}", 2],
                      denoise=1.0)
        frames = node(f"decode{i}", "VAEDecode", samples=latent, vae=vae)
        previous = frames
        if i == 0:
            acc = frames
        else:
            tail = node(f"tail{i}", "ImageFromBatch", image=frames, batch_index=OVERLAP, length=p.plan.length - OVERLAP)
            acc = node(f"cat{i}", "ImageBatch", image1=acc, image2=tail)
    node("save", "SaveImage", images=acc, filename_prefix=p.output_prefix)
    return g
