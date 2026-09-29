"""Exercise startup weight validation without importing the GPU inference stack."""
import ast
import os
from pathlib import Path

import pytest


def validator(tmp_path):
    source = ast.parse((Path(__file__).parents[1] / 'predict.py').read_text())
    function = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == 'ensure_weights')
    namespace = {'os': os, 'COMFY_DIR': str(tmp_path), 'WEIGHTS': {'vae/test.bin': ('unused-build-url', 4)}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'predict.py', 'exec'), namespace)
    return namespace['ensure_weights']


def test_missing_weight_fails_without_download(tmp_path):
    with pytest.raises(RuntimeError, match='Missing bundled weight'):
        validator(tmp_path)()


def test_truncated_weight_fails(tmp_path):
    path = tmp_path / 'models/vae/test.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'bad')
    with pytest.raises(RuntimeError, match='expected 4 bytes, got 3'):
        validator(tmp_path)()


def test_complete_weight_passes(tmp_path):
    path = tmp_path / 'models/vae/test.bin'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'good')
    validator(tmp_path)()
