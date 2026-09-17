"""Minimal stand-in for the `cog` package so predict.py can be exercised on a
GPU box without installing cog/docker. Only what predict.py uses."""
import pathlib
from pydantic import BaseModel  # noqa: F401  (re-exported)


class Path(type(pathlib.Path())):
    pass


class BasePredictor:
    def setup(self):
        pass


def Input(default=None, **_kw):
    return default
