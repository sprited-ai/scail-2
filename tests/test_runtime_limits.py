import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hung_worker_is_terminated():
    result = subprocess.run([sys.executable, "-c", '''
import time
from runtime_limits import deadline
class Worker:
    def predict(self):
        with deadline(0.1, "predict"):
            time.sleep(60)
Worker().predict()
'''], cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 124
    assert "predict exceeded" in result.stdout


def test_completed_call_disarms_timer():
    result = subprocess.run([sys.executable, "-c", '''
import time
from runtime_limits import deadline
class Worker:
    def predict(self):
        with deadline(0.1, "predict"):
            return 42
assert Worker().predict() == 42
time.sleep(0.2)
'''], cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
