import numpy as np
import pytest
from PIL import Image

import preprocess as pp


def test_round_frames():
    assert pp.round_frames_down(65) == 65
    assert pp.round_frames_down(160) == 157
    assert pp.round_frames_down(3) == 1
    assert pp.round_frames_up(83) == 85
    assert pp.round_frames_up(81) == 81


def test_target_size_keeps_aspect_and_step():
    assert pp.target_size(512, 512) == (512, 512)
    assert pp.target_size(1280, 720, "512p") == (896, 512)       # fal's 512p landscape
    assert pp.target_size(720, 1280, "512p") == (512, 896)
    assert pp.target_size(1920, 1080, "704p") == (1248, 704)
    assert pp.target_size(640, 640, "704p") == (704, 704)
    assert pp.target_size(100, 100, width=640, height=384) == (640, 384)
    with pytest.raises(ValueError):
        pp.target_size(100, 100, width=500, height=512)
    with pytest.raises(ValueError):
        pp.target_size(100, 100, width=512)


def test_select_frames_source_rate_and_cap():
    assert pp.select_frames(65, 24, 0, 161) == list(range(65))
    assert len(pp.select_frames(300, 24, 0, 161)) == 161
    assert len(pp.select_frames(64, 24, 0, 161)) == 61            # trimmed to 4k+1


def test_select_frames_resamples_by_holding():
    # 40 fps source of 3 frames per drawing -> 24 fps: nearest source frame, never interpolated
    idx = pp.select_frames(120, 40, 24, 161)
    assert len(idx) == 69                                        # floor(3s * 24) = 72 -> 69 (4k+1)
    assert idx[:4] == [0, 2, 3, 5]
    assert all(b >= a for a, b in zip(idx, idx[1:]))
    # 16 fps source to 24 fps holds frames (duplicates), still monotonic
    up = pp.select_frames(48, 16, 24, 161)
    assert len(up) == 69 and up[-1] == 45 and max(up) < 48


def test_map_indices():
    assert pp.map_indices([0, 2, 4], 10, 10) == [0, 2, 4]
    assert pp.map_indices([0, 5, 9], 10, 20) == [0, 10, 18]
    assert pp.map_indices([0, 5, 9], 10, 5) == [0, 2, 4]


def test_plan_chunks_single_window():
    plan = pp.plan_chunks(65)
    assert plan.length == 65 and plan.starts == [0] and plan.total == 65 and plan.offsets == [0]


def test_plan_chunks_two_windows_cover_exactly():
    plan = pp.plan_chunks(157)  # 81 + 76 = fal's 160-frame cap after 4k+1 rounding
    assert plan.length == 81 and plan.starts == [0, 76] and plan.total == 157
    assert plan.offsets == [0, 81]  # node subtracts the 5 anchor frames -> slices pose video at 76


def test_plan_chunks_balances_instead_of_tiny_tail():
    plan = pp.plan_chunks(161)
    assert plan.length == 57 and plan.starts == [0, 52, 104] and plan.total == 161
    assert all(l % 4 == 1 for l in [plan.length])
    for n in range(85, pp.MAX_FRAMES + 1, 4):
        p = pp.plan_chunks(n)
        assert p.length <= pp.CHUNK and p.length % 4 == 1 and p.total >= n and p.total - n < p.length
        assert p.starts == [i * (p.length - pp.OVERLAP) for i in range(len(p.starts))]


def test_assemble_drops_overlap_and_trims():
    a = np.arange(81)[:, None]
    b = np.arange(76, 157)[:, None]  # window 2 starts 5 frames before window 1 ends
    out = pp.assemble([a, b], 157)
    assert out.shape[0] == 157 and out[:, 0].tolist() == list(range(157))
    assert pp.assemble([a, b], 100).shape[0] == 100


def test_colorize_and_snap():
    m = np.zeros((4, 4), dtype=bool); m[1:3, 1:3] = True
    anim = pp.colorize(m, "black")
    assert tuple(anim[0, 0]) == pp.BLACK and tuple(anim[1, 1]) == pp.IDENTITY
    rep = pp.colorize(m, "white")
    assert tuple(rep[0, 0]) == pp.WHITE
    noisy = anim.astype(np.int16) + np.array([3, -2, -20]); noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    assert np.array_equal(pp.snap_palette(noisy), anim)


def test_mask_from_grayscale_matte_is_colored_on_mode_background():
    matte = Image.fromarray(np.uint8(np.pad(np.full((2, 2), 255), 1)))  # 4x4, white centre
    out = pp.mask_from_image(matte, "white", 64, 64, "letterbox")
    assert out.shape == (64, 64, 3)
    assert tuple(out[0, 0]) == pp.WHITE and tuple(out[32, 32]) == pp.IDENTITY
    out = pp.mask_from_image(matte, "black", 64, 64, "cover")
    assert tuple(out[0, 0]) == pp.BLACK and tuple(out[32, 32]) == pp.IDENTITY


def test_mask_from_colored_image_passes_through_snapped():
    src = pp.colorize(np.eye(8, dtype=bool), "white")
    src[0, 0] = (250, 250, 250)  # slightly off-white survives snapping
    out = pp.mask_from_image(Image.fromarray(src), "white", 8, 8, "letterbox")
    assert tuple(out[0, 0]) == pp.WHITE and tuple(out[3, 3]) == pp.IDENTITY and tuple(out[0, 7]) == pp.WHITE


def test_mask_from_alpha_and_flatten():
    rgba = np.zeros((10, 20, 4), dtype=np.uint8); rgba[..., :3] = 200; rgba[2:8, 5:15, 3] = 255
    rgb, alpha = pp.flatten_alpha(Image.fromarray(rgba, "RGBA"))
    assert rgb.mode == "RGB" and tuple(np.asarray(rgb)[0, 0]) == pp.WHITE and tuple(np.asarray(rgb)[5, 10]) == (200, 200, 200)
    m = pp.mask_from_alpha(alpha, "white", 64, 32)
    assert m.shape == (32, 64, 3) and tuple(m[16, 32]) == pp.IDENTITY and tuple(m[0, 0]) == pp.WHITE


def test_letterbox_and_cover_geometry():
    img = Image.new("RGB", (100, 50), (10, 20, 30))
    lb = pp.letterbox(img, 64, 64, pp.WHITE)
    assert lb.size == (64, 64) and tuple(np.asarray(lb)[0, 0]) == pp.WHITE and tuple(np.asarray(lb)[32, 32]) == (10, 20, 30)
    cv = pp.cover(img, 64, 64)
    assert cv.size == (64, 64) and tuple(np.asarray(cv)[0, 0]) == (10, 20, 30)
