"""Pure input preparation for the SCAIL-2 predictor: sizing, frame selection,
chunk planning and SCAIL-2 mask colouring. No torch, no ComfyUI — unit-testable
on a laptop (see tests/).

SCAIL-2 mask conventions (from the paper / ComfyUI nodes_scail.py):
  * identity colours come from a fixed palette the model was trained on; the
    first identity is pure blue.
  * driving-video mask: black background in animation mode (the background in
    the driving video is *not* shown — it comes from the reference), white
    background in replacement mode (the driving video's background is kept).
  * reference mask: the opposite — white background in animation mode, black
    in replacement mode.
  * ComfyUI thresholds every channel at 225/255, so colours must be saturated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Fixed SCAIL-2 identity palette; index 0 (blue) is the single-subject colour.
PALETTE = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
IDENTITY = PALETTE[0]
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ON_THRESHOLD = 225  # ComfyUI: channel > 225/255 counts as "on"

RESOLUTIONS = {"512p": 512, "704p": 704}  # short side
STEP = 32                                  # WanSCAILToVideo width/height step
CHUNK = 81                                 # frames per window the model was trained on
OVERLAP = 5                                # previous_frame_count anchor (Comfy default)
MAX_FRAMES = 161                           # hard cap per prediction (2 windows of 81)


# ----------------------------------------------------------------------------
# frame counts / sizes
# ----------------------------------------------------------------------------

def round_frames_down(n: int) -> int:
    """Largest 4k+1 <= n (Wan latent packs 4 frames per latent frame + 1)."""
    return max(1, ((n - 1) // 4) * 4 + 1)


def round_frames_up(n: int) -> int:
    """Smallest 4k+1 >= n."""
    return max(1, ((n - 1 + 3) // 4) * 4 + 1)


def round_to_step(v: float, step: int = STEP) -> int:
    return max(step, int(round(v / step)) * step)


def target_size(src_w: int, src_h: int, resolution: str = "512p", width: int = 0, height: int = 0) -> tuple[int, int]:
    """Output (W, H), both multiples of 32.

    Explicit width+height win. Otherwise the driving video's aspect ratio is
    kept and its short side is set by `resolution` (512p -> 512, 704p -> 704),
    so a 16:9 driver at 512p becomes 896x512 and a square one 512x512.
    """
    if width or height:
        if not (width and height):
            raise ValueError("set both width and height (multiples of 32), or neither")
        if width % STEP or height % STEP:
            raise ValueError(f"width and height must be multiples of {STEP} (got {width}x{height})")
        return width, height
    short = RESOLUTIONS[resolution]
    if src_w >= src_h:
        return round_to_step(short * src_w / src_h), short
    return short, round_to_step(short * src_h / src_w)


def select_frames(n_src: int, src_fps: float, out_fps: float, max_frames: int) -> list[int]:
    """Source frame indices to keep: resampled to `out_fps` by nearest-frame
    (frames are held, never interpolated), capped at `max_frames`, and trimmed
    to a 4k+1 count. out_fps <= 0 keeps the source rate."""
    if n_src <= 0:
        return []
    if out_fps <= 0 or src_fps <= 0 or abs(out_fps - src_fps) < 1e-3:
        idx = list(range(n_src))
    else:
        n_out = int(math.floor(n_src / src_fps * out_fps + 1e-6))
        idx = [min(n_src - 1, int(i * src_fps / out_fps + 0.5)) for i in range(max(n_out, 1))]
    idx = idx[:max_frames]
    return idx[:round_frames_down(len(idx))]


def map_indices(selected: list[int], n_src: int, n_other: int) -> list[int]:
    """Map frame indices chosen on an n_src-frame video onto an n_other-frame
    companion (a mask video). Identical lengths map 1:1; otherwise by time."""
    if n_other == n_src:
        return [min(i, n_other - 1) for i in selected]
    return [min(n_other - 1, int(i * n_other / max(n_src, 1))) for i in selected]


@dataclass(frozen=True)
class ChunkPlan:
    length: int        # frames per window (4k+1, <= CHUNK); every window is the same length
    starts: list[int]  # first output frame of each window
    total: int         # frames the windows cover together (>= requested); pad the driver to this

    @property
    def offsets(self) -> list[int]:
        """`video_frame_offset` to pass to WanSCAILToVideo per window: the node
        subtracts the OVERLAP anchor frames itself for every window after the first."""
        return [0] + [s + OVERLAP for s in self.starts[1:]]


def plan_chunks(n_frames: int, chunk: int = CHUNK, overlap: int = OVERLAP) -> ChunkPlan:
    """Split n_frames (4k+1) into equal windows the node can generate. Windows
    after the first are re-anchored on the previous window's last `overlap`
    frames, so k windows of length L cover k*L - (k-1)*overlap frames. Equal
    lengths keep every window well inside the trained range instead of
    leaving a tiny tail window."""
    if n_frames <= chunk:
        return ChunkPlan(n_frames, [0], n_frames)
    k = 2
    while True:
        length = round_frames_up(math.ceil((n_frames + (k - 1) * overlap) / k))
        if length <= chunk:
            break
        k += 1
    starts = [i * (length - overlap) for i in range(k)]
    return ChunkPlan(length, starts, starts[-1] + length)


def assemble(chunks: list[np.ndarray], n_frames: int, overlap: int = OVERLAP) -> np.ndarray:
    """Concatenate decoded windows, dropping each later window's `overlap`
    anchor frames (they duplicate the previous window's tail)."""
    parts = [chunks[0]] + [c[overlap:] for c in chunks[1:]]
    return np.concatenate(parts, axis=0)[:n_frames]


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------

def flatten_alpha(img: Image.Image, fill=WHITE) -> tuple[Image.Image, np.ndarray | None]:
    """RGB image composited over `fill`, plus the alpha channel (uint8) if any."""
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A"))
        bg = Image.new("RGBA", rgba.size, fill + (255,))
        return Image.alpha_composite(bg, rgba).convert("RGB"), alpha
    return img.convert("RGB"), None


def letterbox(img: Image.Image, w: int, h: int, fill, resample=Image.LANCZOS) -> Image.Image:
    """Fit inside w x h keeping aspect (no crop), pad with `fill`."""
    sw, sh = img.size
    scale = min(w / sw, h / sh)
    nw, nh = max(1, round(sw * scale)), max(1, round(sh * scale))
    canvas = Image.new(img.mode, (w, h), fill)
    canvas.paste(img.resize((nw, nh), resample), ((w - nw) // 2, (h - nh) // 2))
    return canvas


def cover(img: Image.Image, w: int, h: int, resample=Image.LANCZOS) -> Image.Image:
    """Fill w x h keeping aspect, centre-cropping the excess."""
    sw, sh = img.size
    scale = max(w / sw, h / sh)
    nw, nh = max(w, round(sw * scale)), max(h, round(sh * scale))
    img = img.resize((nw, nh), resample)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))


# ----------------------------------------------------------------------------
# masks
# ----------------------------------------------------------------------------

def colorize(mask: np.ndarray, background: str, color=IDENTITY) -> np.ndarray:
    """Boolean HxW subject mask -> HxWx3 uint8 SCAIL mask (subject in `color`)."""
    bg = WHITE if background == "white" else BLACK
    out = np.empty(mask.shape + (3,), dtype=np.uint8)
    out[...] = bg
    out[mask] = color
    return out


def snap_palette(rgb: np.ndarray) -> np.ndarray:
    """Binarise each channel at the ComfyUI threshold so colours survive
    resizing/compression exactly as the model will read them."""
    return np.where(rgb > ON_THRESHOLD, 255, 0).astype(np.uint8)


def is_colored_mask(rgb: np.ndarray) -> bool:
    """True if the image already uses SCAIL identity colours (any pixel whose
    channels differ), False for a plain grayscale/binary matte."""
    rgb = rgb.astype(np.int16)
    return bool(np.any(np.abs(rgb[..., 0] - rgb[..., 1]) > 40) or np.any(np.abs(rgb[..., 1] - rgb[..., 2]) > 40))


def mask_from_image(img: Image.Image, background: str, w: int, h: int, fit: str) -> np.ndarray:
    """A user-supplied mask image -> HxWx3 SCAIL mask at (w, h).

    Grayscale / binary mattes (white subject) get the identity colour on the
    mode's background; images already in palette colours pass through (snapped
    to exact palette values). `fit` is 'letterbox' (reference) or 'cover' (driving)."""
    rgb, alpha = flatten_alpha(img, BLACK)
    arr = np.asarray(rgb)
    if is_colored_mask(arr):
        fill = WHITE if background == "white" else BLACK
        resized = letterbox(rgb, w, h, fill, Image.NEAREST) if fit == "letterbox" else cover(rgb, w, h, Image.NEAREST)
        return snap_palette(np.asarray(resized))
    gray = np.asarray(alpha if alpha is not None and arr.max() == 0 else rgb.convert("L"))
    mask_img = Image.fromarray(gray)
    resized = letterbox(mask_img, w, h, 0, Image.BILINEAR) if fit == "letterbox" else cover(mask_img, w, h, Image.BILINEAR)
    return colorize(np.asarray(resized) >= 128, background)


def mask_from_alpha(alpha: np.ndarray, background: str, w: int, h: int) -> np.ndarray:
    """Alpha channel of a cutout reference (already letterboxed to w x h) -> SCAIL mask."""
    a = Image.fromarray(alpha)
    if a.size != (w, h):
        a = letterbox(a, w, h, 0, Image.BILINEAR)
    return colorize(np.asarray(a) >= 128, background)


def blank_mask(background: str, w: int, h: int) -> np.ndarray:
    return colorize(np.zeros((h, w), dtype=bool), background)


# ----------------------------------------------------------------------------
# runtime estimate (Replicate kills predictions at 30 minutes)
# ----------------------------------------------------------------------------

WAN_PARAMS = 14e9          # SCAIL-14B (Wan 2.1 14B DiT)
WAN_HIDDEN, WAN_LAYERS = 5120, 40
PATCH_TOKENS = 8 * 8 * 2 * 2   # pixels per token: VAE x8 spatial, then 2x2 patch embedding


def latent_tokens(width: int, height: int, frames: int) -> int:
    return (((frames - 1) // 4) + 1) * (width * height // PATCH_TOKENS)


def estimate_seconds(width: int, height: int, plan: ChunkPlan, steps: int, cfg: float,
                     effective_tflops: float = 350.0) -> float:
    """Rough sampling-time estimate: per forward pass, 2*params*tokens FLOPs for the
    linear layers plus 4*tokens^2*hidden*layers for full self-attention, divided
    by an assumed sustained throughput. CFG > 1 doubles the passes. Calibrate
    `effective_tflops` against measured runs (an RTX PRO 6000 without fp8
    matmul measured ~13 s/pass at 81 frames 896x512, i.e. ~170 TFLOPS)."""
    tokens = latent_tokens(width, height, plan.length)
    flops = 2 * WAN_PARAMS * tokens + 4 * tokens ** 2 * WAN_HIDDEN * WAN_LAYERS
    passes = steps * (2 if cfg > 1 else 1) * len(plan.starts)
    return passes * flops / (effective_tflops * 1e12)
