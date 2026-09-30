import ast
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

import preprocess as pp
import workflow as wf
from test_workflow import params, check_against_schema


def test_sprute_graph_uses_local_sampler_model_and_pose_controls(monkeypatch):
    from test_workflow import OI
    # Loader dropdowns are generated from the installed files, not a fixed API enum.
    required = OI["VAELoader"]["input"]["required"]
    monkeypatch.setitem(required, "vae_name", [[wf.VAE, wf.VAE_BF16]])
    pr = wf.PRESETS['balanced']
    g = wf.build_graph(params(steps=pr['steps'], cfg=pr['cfg'], shift=pr['shift'],
                             sampler=pr['sampler'], unet=wf.UNET, vae=wf.VAE_BF16,
                             pose_start=.2, pose_end=.8, denoise=.75,
                             loras=[(wf.LORA_DPO,1), (wf.LORA_LIGHTX2V,.8)]))
    check_against_schema(g)
    assert g['unet']['inputs']['unet_name'] == wf.UNET
    assert g['vae']['inputs']['vae_name'] == wf.VAE_BF16
    assert g['sample0']['inputs']['steps'] == 8
    assert g['sample0']['inputs']['sampler_name'] == 'uni_pc'
    assert g['sample0']['inputs']['denoise'] == .75
    assert g['scail0']['inputs']['pose_start'] == .2
    assert g['scail0']['inputs']['pose_end'] == .8


def test_prepared_grid_and_palette_are_bit_exact():
    grid = np.random.default_rng(42).integers(0,256,(64,96,3),dtype=np.uint8)
    assert np.array_equal(pp.prepared_rgb(Image.fromarray(grid),96,64),grid)
    palette = np.array([[[0,0,0],[255,255,255],[0,0,255],[255,0,0]]],dtype=np.uint8)
    assert np.array_equal(pp.prepared_rgb(Image.fromarray(palette),4,1,palette=True),palette)
    with pytest.raises(ValueError,match='dimensions'):
        pp.prepared_rgb(Image.fromarray(grid),64,64)
    with pytest.raises(ValueError,match='opaque'):
        pp.prepared_rgb(Image.new('RGBA',(96,64),(1,2,3,128)),96,64)
    with pytest.raises(ValueError,match='palette'):
        pp.prepared_rgb(Image.new('RGB',(4,1),(0,0,254)),4,1,palette=True)


def test_frame_archive_preserves_order_and_original_png_bytes(tmp_path):
    paths=[]
    for i in range(3):
        p=tmp_path/f'f_{i+1:05d}_.png'
        Image.new('RGB',(32,32),(i*80,10,20)).save(p)
        paths.append(p)
    dest=tmp_path/'frames.zip'; pp.archive_frames(paths,dest)
    with zipfile.ZipFile(dest) as z:
        assert z.namelist() == ['000000.png','000001.png','000002.png']
        for n,p in zip(z.namelist(),paths): assert z.read(n)==p.read_bytes()


def video_helpers():
    import av
    tree=ast.parse(Path('predict.py').read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_webp','probe_video','decode_selected')]
    ns={'av':av,'Image':Image,'MAX_PROBE_FRAMES':20000}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'predict.py','exec'),ns)
    return ns


def test_lossless_webp_decodes_exact_rgb_and_timing(tmp_path):
    h=video_helpers(); p=tmp_path/'driver.bin'
    frames=[Image.new('RGB',(64,64),(i*40,128,255)) for i in range(5)]
    frames[0].save(p,format='WEBP',save_all=True,append_images=frames[1:],duration=40,lossless=True)
    assert h['probe_video'](str(p)) == (5,25.0,64,64)
    decoded=h['decode_selected'](str(p),[0,2,4],lambda im:np.asarray(im.convert('RGB')))
    for a,i in zip(decoded,[0,2,4]):assert np.array_equal(a,np.asarray(frames[i]))


@pytest.mark.parametrize("extended", [False, True])
def test_predictor_prepared_request_reaches_graph_and_returns_lossless_frames(tmp_path, extended):
    """Exercise the real request path; replace only the GPU executor and MP4 encoder."""
    import contextlib
    import glob
    import os
    import random
    import shutil
    import time
    import types
    import uuid
    from fractions import Fraction
    from typing import Optional
    import av

    tree=ast.parse(Path('predict.py').read_text())
    predictor=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Predictor')
    method=next(n for n in predictor.body if isinstance(n,ast.FunctionDef) and n.name=='predict')
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('is_webp','probe_video','decode_selected','write_video')]
    inp=tmp_path/'input';out=tmp_path/'output';inp.mkdir();out.mkdir()
    class Output:
        def __init__(self,**kw):self.__dict__.update(kw)
    ns=dict(pp=pp,np=np,Image=Image,av=av,Path=Path,Optional=Optional,Output=Output,
            Input=lambda default=None,**kw:default,deadline=lambda *a:contextlib.nullcontext(),
            os=os,time=time,random=random,uuid=uuid,glob=glob,shutil=shutil,json=json,Fraction=Fraction,
            INPUT_DIR=str(inp),OUTPUT_DIR=str(out),WORK=str(tmp_path),MAX_PROBE_FRAMES=20000,
            MAX_SAMPLING_SECONDS=720,EFFECTIVE_TFLOPS=350,
            encode_mp4=lambda pattern,n,fps,p:Path(p).write_bytes(b'preview'))
    ns.update({name:getattr(wf,name) for name in ('PRESETS','LORA_DPO','LORA_RELIGHT','LORA_LIGHTX2V','UNET','VAE','VAE_BF16','GraphParams','build_graph')})
    exec(compile(ast.Module(body=helpers+[method],type_ignores=[]),'predict.py','exec'),ns)
    pixels=np.full((64,96,3),128,dtype=np.uint8)
    pixels[16:48,32:64]=(10,20,30)
    mask=np.zeros_like(pixels);mask[16:48,32:64]=(0,0,255)
    ref=tmp_path/'ref.png';Image.fromarray(pixels).save(ref)
    rm=tmp_path/'rm.png';Image.fromarray(mask).save(rm)
    drive=tmp_path/'drive.mkv';dm=tmp_path/'mask.mkv'
    ns['write_video'](str(drive),[pixels]*5,24)
    ns['write_video'](str(dm),[mask]*5,24)
    class Comfy:
        def alive(self):return True
        def run(self,graph,timeout):
            self.graph=graph
            assert np.array_equal(np.asarray(Image.open(inp/graph['ref']['inputs']['image'])),pixels)
            assert np.array_equal(np.asarray(Image.open(inp/graph['ref_mask']['inputs']['image'])),mask)
            maskfile=inp/graph['drive_mask']['inputs']['file']
            decoded=ns['decode_selected'](str(maskfile),[0,4],lambda im:np.asarray(im))
            assert all(np.array_equal(x,mask) for x in decoded)
            prefix=out/graph['save']['inputs']['filename_prefix'];prefix.parent.mkdir()
            for i in range(5):Image.fromarray(pixels).save(str(prefix)+f'_{i+1:05d}_.png')
    comfy=Comfy();owner=types.SimpleNamespace(comfy=comfy)
    extra = dict(additional_images=[ref], additional_image_masks=[rm], previous_frames=ref,
                 previous_frame_count=1) if extended else {}
    result=ns['predict'](owner,image=ref,video=drive,image_mask=rm,video_mask=dm,
                         prepared_inputs=True,preset='balanced',vae_precision='bf16',
                         seed=42,return_frames=True,lightx2v_lora=.6,pose_start=.1,pose_end=.9,denoise=.8, **extra)
    meta=json.loads(result.metadata.read_text())
    assert (meta['width'],meta['height'],meta['num_frames'],meta['fps'])==(96,64,5,24)
    assert meta['sampler']=='uni_pc' and meta['steps']==8
    assert meta['model']==wf.UNET and meta['vae']==wf.VAE_BF16
    assert meta['loras']==[[wf.LORA_DPO,1.0],[wf.LORA_LIGHTX2V,.6]]
    assert comfy.graph['scail0']['inputs']['pose_start']==.1
    assert comfy.graph['sample0']['inputs']['denoise']==.8
    if extended:
        assert comfy.graph['scail0']['inputs']['reference_image'] == ['ref_batch0', 0]
        assert comfy.graph['scail0']['inputs']['reference_image_mask'] == ['mask_batch0', 0]
        assert comfy.graph['scail0']['inputs']['previous_frames'] == ['previous_images', 0]
        assert comfy.graph['scail0']['inputs']['previous_frame_count'] == 1
        assert meta['reference_count'] == 2 and meta['previous_frames_supplied']
    with zipfile.ZipFile(result.frames) as z:
        from io import BytesIO
        assert len(z.namelist())==5
        assert np.array_equal(np.asarray(Image.open(BytesIO(z.read('000004.png')))),pixels)
    assert not list(inp.iterdir())


@pytest.mark.parametrize("overlap", [1, 5, 9, 77])
def test_continuation_overlap_is_consistent(overlap):
    plan = pp.plan_chunks(161, overlap=overlap)
    graph = wf.build_graph(params(plan=plan, previous_frames='previous.mkv'))
    check_against_schema(graph)
    for i, start in enumerate(plan.starts):
        cond = graph[f'scail{i}']['inputs']
        assert cond['previous_frame_count'] == overlap
        assert cond['video_frame_offset'] == (start + overlap if i else 0)
        if i:
            assert graph[f'tail{i}']['inputs']['batch_index'] == overlap
    assert plan.total >= 161
    for invalid in [0, 2, 81]:
        with pytest.raises(ValueError):
            pp.plan_chunks(161, overlap=invalid)

@pytest.mark.parametrize('auto_mask,explicit,transparent', [(True,False,True),(False,False,True),(True,True,True),(True,False,False),(False,False,False)])
def test_driving_webp_alpha_mask_priority(tmp_path, auto_mask, explicit, transparent):
    """Execute the actual decode/mask branch without loading GPU dependencies."""
    import time
    from types import SimpleNamespace
    path=tmp_path/'driver.webp'
    source=[]
    for i in range(5):
        im=Image.new('RGBA',(96,64),(255,0,0,0 if transparent else 255))
        im.paste((10,100+i,200,255),(24,16,72,48));source.append(im)
    source[0].save(path,save_all=True,append_images=source[1:],duration=40,lossless=True)
    supplied=tmp_path/'mask.webp'
    masks=[Image.new('RGB',(96,64),(i, i, i)) for i in range(5)]
    masks[0].save(supplied,save_all=True,append_images=masks[1:],duration=40,lossless=True)
    class Matter:
        calls=0
        def masks(self, frames):
            self.calls+=1
            return [np.zeros((64,64),dtype=bool) for _ in frames]
    matter=Matter()
    ns=video_helpers()
    ns.update(pp=pp,np=np,time=time,video=path,sel=[0,2,2,4],W=64,H=64,n=4,n_src=5,src_fps=25,
              prepared_inputs=False,video_mask=supplied if explicit else None,auto_mask=auto_mask,
              drive_bg='black',self=SimpleNamespace(matter=matter))
    tree=ast.parse(Path('predict.py').read_text())
    body=next(node.body for node in ast.walk(tree) if isinstance(node,ast.With) and any(isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='decoded' for x in node.body))
    start=next(i for i,x in enumerate(body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='decoded')
    end=next(i for i,x in enumerate(body) if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Tuple) and isinstance(x.targets[0].elts[0],ast.Name) and x.targets[0].elts[0].id=='ref_rgb')
    exec(compile(ast.Module(body=body[start:end],type_ignores=[]),'predict.py','exec'),ns)
    assert len(ns['frames'])==4
    assert matter.calls==int(auto_mask and not explicit and not transparent)
    if explicit:
        assert not np.any(ns['drive_mask'])
    elif transparent:
        for mask in ns['drive_mask']:
            assert tuple(mask[32,32])==(0,0,255)
            assert tuple(mask[0,0])==(0,0,0)
        assert np.array_equal(ns['drive_mask'][1],ns['drive_mask'][2])
    elif not auto_mask:
        assert ns['drive_mask'] is None
