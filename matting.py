"""BiRefNet (MIT, ZhengPeng7) single-subject matting used to derive SCAIL-2
masks when the caller supplies none. Same model as sprited/birefnet."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

REPO = "ZhengPeng7/BiRefNet"
RES = 1024
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class BiRefNetMatter:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForImageSegmentation
        print("[matting] loading BiRefNet", flush=True)
        try:  # baked HF cache first; only touch the Hub if it is missing
            model = AutoModelForImageSegmentation.from_pretrained(REPO, trust_remote_code=True, local_files_only=True)
        except Exception as e:  # noqa: BLE001
            print(f"[matting] cache miss ({type(e).__name__}); downloading", flush=True)
            model = AutoModelForImageSegmentation.from_pretrained(REPO, trust_remote_code=True)
        model = model.to(self.device).eval()
        self.half = self.device == "cuda"
        if self.half:
            model = model.half()
        self.model = model
        self.mean, self.std = MEAN.to(self.device), STD.to(self.device)

    @torch.no_grad()
    def masks(self, frames: list[np.ndarray], batch: int = 8, threshold: float = 0.5) -> list[np.ndarray]:
        """HxWx3 uint8 frames -> boolean HxW subject masks (same size as input)."""
        self.load()
        out: list[np.ndarray] = []
        for i in range(0, len(frames), batch):
            chunk = np.stack(frames[i:i + batch])                      # B,H,W,3
            h, w = chunk.shape[1:3]
            x = torch.from_numpy(chunk).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
            x = (x - self.mean) / self.std
            if self.half:
                x = x.half()
            pred = self.model(x)[-1].sigmoid().float()                # B,1,RES,RES
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            out.extend((pred[:, 0] >= threshold).cpu().numpy())
        return out
