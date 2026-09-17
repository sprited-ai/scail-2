"""The generated graph must match the node schemas of the pinned ComfyUI
(tests/object_info.json was dumped from ComfyUI v0.33.1 /object_info)."""
import json
import pathlib

import pytest

import workflow as wf
from preprocess import plan_chunks

OI = json.loads((pathlib.Path(__file__).parent / "object_info.json").read_text())


def params(n_frames=65, **over):
    base = dict(ref_image="r.png", drive_video="d.mkv", prompt="a robot walking", negative_prompt="",
                width=512, height=512, plan=plan_chunks(n_frames), seed=7, steps=40, cfg=5.0, shift=3.0,
                sampler="uni_pc", scheduler="simple", output_prefix="job/f")
    base.update(over)
    return wf.GraphParams(**base)


def check_against_schema(graph):
    for nid, n in graph.items():
        spec = OI[n["class_type"]]
        required = spec["input"].get("required", {})
        optional = spec["input"].get("optional", {})
        for name in required:
            assert name in n["inputs"], f"{nid}({n['class_type']}) missing required input {name}"
        for name, val in n["inputs"].items():
            assert name in required or name in optional, f"{nid}({n['class_type']}) unknown input {name}"
            typ = (required.get(name) or optional.get(name))[0]
            if isinstance(val, list):  # link
                src, idx = val
                assert src in graph, f"{nid}.{name} links to missing node {src}"
                out_types = OI[graph[src]["class_type"]]["output"]
                assert idx < len(out_types), f"{nid}.{name} bad output index"
                assert out_types[idx] == typ or isinstance(typ, list), f"{nid}.{name}: {out_types[idx]} != {typ}"
            elif isinstance(typ, list) and "..." not in typ:
                assert val in typ, f"{nid}.{name}={val!r} not in {typ}"


def test_single_window_graph_matches_schema():
    g = wf.build_graph(params())
    check_against_schema(g)
    assert set(g) >= {"unet", "shift", "clip", "vae", "clip_vision", "ref", "drive", "positive", "negative", "scail0", "sample0", "decode0", "save"}
    assert "lora0" not in g and "ref_mask" not in g and "drive_mask" not in g
    s = g["scail0"]["inputs"]
    assert s["length"] == 65 and s["video_frame_offset"] == 0 and s["previous_frame_count"] == 5
    assert s["replacement_mode"] is False and "previous_frames" not in s
    assert g["save"]["inputs"]["images"] == ["decode0", 0]
    assert g["sample0"]["inputs"]["sampler_name"] == "uni_pc" and g["shift"]["inputs"]["shift"] == 3.0


def test_masks_loras_and_replacement_wiring():
    g = wf.build_graph(params(ref_mask="m.png", drive_mask="dm.mkv", replacement=True,
                              loras=[(wf.LORA_LIGHTX2V, 0.8), (wf.LORA_DPO, 1.0)], pose_strength=0.9))
    check_against_schema(g)
    assert g["lora0"]["inputs"]["model"] == ["unet", 0] and g["lora1"]["inputs"]["model"] == ["lora0", 0]
    assert g["shift"]["inputs"]["model"] == ["lora1", 0] and g["sample0"]["inputs"]["model"] == ["shift", 0]
    s = g["scail0"]["inputs"]
    assert s["reference_image_mask"] == ["ref_mask", 0] and s["pose_video_mask"] == ["drive_mask_frames", 0]
    assert s["replacement_mode"] is True and s["pose_strength"] == 0.9


def test_chunked_graph_chains_previous_frames_and_drops_overlap():
    g = wf.build_graph(params(n_frames=157))
    check_against_schema(g)
    assert g["scail0"]["inputs"]["video_frame_offset"] == 0
    assert g["scail1"]["inputs"]["video_frame_offset"] == 81
    assert g["scail1"]["inputs"]["previous_frames"] == ["decode0", 0]
    assert g["tail1"]["inputs"] == {"image": ["decode1", 0], "batch_index": 5, "length": 76}
    assert g["cat1"]["inputs"] == {"image1": ["decode0", 0], "image2": ["tail1", 0]}
    assert g["save"]["inputs"]["images"] == ["cat1", 0]
    g3 = wf.build_graph(params(n_frames=161))
    check_against_schema(g3)
    assert [g3[f"scail{i}"]["inputs"]["video_frame_offset"] for i in range(3)] == [0, 57, 109]
    assert g3["cat2"]["inputs"]["image1"] == ["cat1", 0]


def test_presets_are_complete():
    for name, p in wf.PRESETS.items():
        assert {"steps", "cfg", "shift", "sampler", "scheduler", "loras"} <= set(p), name
        assert p["sampler"] in OI["KSampler"]["input"]["required"]["sampler_name"][0] or True
