"""Worker deadlines; API Cancel-After must separately bound platform startup."""
from contextlib import contextmanager
import os
import threading


@contextmanager
def deadline(seconds: float, operation: str):
    finished = threading.Event()

    def expire():
        if finished.is_set():
            return
        print(f"[timeout] {operation} exceeded {seconds}s; terminating worker", flush=True)
        try:
            # Include weight downloaders and the Comfy process tree, not just Python.
            import psutil
            for child in psutil.Process().children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        finally:
            # A Python exception in this thread cannot interrupt native GPU work.
            os._exit(124)

    timer = threading.Timer(seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        finished.set()
        timer.cancel()
