"""Thin client for a headless ComfyUI: start it as a subprocess, submit an
API-format graph, and follow it to completion over the websocket.

Errors from ComfyUI (graph validation, node exceptions) are re-raised as
ComfyError with the node name and message so they surface in Replicate logs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

import websocket  # websocket-client


class ComfyError(RuntimeError):
    pass


class ComfyServer:
    def __init__(self, comfy_dir: str, input_dir: str, output_dir: str, temp_dir: str,
                 host: str = "127.0.0.1", port: int = 8188, extra_args: tuple[str, ...] = ()):
        self.comfy_dir, self.input_dir, self.output_dir, self.temp_dir = comfy_dir, input_dir, output_dir, temp_dir
        self.host, self.port, self.extra_args = host, port, tuple(extra_args)
        self.base = f"http://{host}:{port}"
        self.proc: subprocess.Popen | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        for d in (self.input_dir, self.output_dir, self.temp_dir):
            os.makedirs(d, exist_ok=True)
        cmd = [sys.executable, "main.py", "--listen", self.host, "--port", str(self.port),
               "--disable-auto-launch", "--disable-all-custom-nodes", "--disable-metadata",
               "--input-directory", self.input_dir, "--output-directory", self.output_dir,
               "--temp-directory", self.temp_dir, *self.extra_args]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        print(f"[comfy] starting: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(cmd, cwd=self.comfy_dir, env=env)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def wait_ready(self, timeout: float = 900) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.alive():
                raise ComfyError(f"ComfyUI exited with code {self.proc.returncode} during startup")
            try:
                self.get("/system_stats")
                print("[comfy] ready", flush=True)
                return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(1)
        raise ComfyError(f"ComfyUI did not become ready within {timeout}s")

    def stop(self) -> None:
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # -- http ----------------------------------------------------------------
    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=60) as r:
            return json.load(r)

    def post(self, path: str, payload: dict):
        req = urllib.request.Request(self.base + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                raise ComfyError(_format_validation(json.loads(body))) from None
            except (ValueError, KeyError):
                raise ComfyError(f"ComfyUI {path} -> HTTP {e.code}: {body[:2000]}") from None

    def object_info(self) -> dict:
        return self.get("/object_info")

    # -- execution -----------------------------------------------------------
    def run(self, graph: dict, timeout: float = 3600, log=print) -> dict:
        """Queue `graph`, stream progress to `log`, return the history outputs."""
        client_id = uuid.uuid4().hex
        ws = websocket.create_connection(f"ws://{self.host}:{self.port}/ws?clientId={client_id}", timeout=60)
        ws.settimeout(30)
        try:
            resp = self.post("/prompt", {"prompt": graph, "client_id": client_id})
            if "error" in resp:
                raise ComfyError(_format_validation(resp))
            prompt_id = resp["prompt_id"]
            deadline = time.time() + timeout
            while True:
                if not self.alive():
                    raise ComfyError("ComfyUI process died during execution")
                if time.time() > deadline:
                    raise ComfyError(f"execution exceeded {timeout}s")
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if isinstance(msg, bytes):
                    continue  # binary previews
                ev = json.loads(msg)
                t, d = ev.get("type"), ev.get("data") or {}
                pid = d.get("prompt_id")
                if pid is not None and pid != prompt_id:
                    continue
                if t == "progress":
                    log(f"[comfy] {d.get('node')} step {d.get('value')}/{d.get('max')}")
                elif t == "executing":
                    if d.get("node") is None and pid == prompt_id:
                        break
                    log(f"[comfy] executing {d.get('node')}")
                elif t == "execution_success" and pid == prompt_id:
                    break
                elif t == "execution_error":
                    tb = "\n".join(d.get("traceback", [])[-6:])
                    raise ComfyError(f"{d.get('node_type')} ({d.get('node_id')}): {d.get('exception_message')}\n{tb}")
                elif t == "execution_interrupted":
                    raise ComfyError("execution interrupted")
        finally:
            ws.close()
        hist = self.get(f"/history/{prompt_id}").get(prompt_id)
        if not hist:
            raise ComfyError("no history entry after execution")
        status = hist.get("status") or {}
        if status.get("status_str") == "error":
            raise ComfyError(f"execution failed: {json.dumps(status.get('messages'))[:2000]}")
        return hist.get("outputs", {})


def _format_validation(resp: dict) -> str:
    err = resp.get("error") or {}
    lines = [f"graph rejected: {err.get('message', err)} {err.get('details', '')}".strip()]
    for nid, ne in (resp.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            lines.append(f"  node {nid} ({ne.get('class_type')}): {e.get('message')} {e.get('details', '')}".rstrip())
    return "\n".join(lines)
