"""The file-size hook passes on this repo."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_check_file_size_cli():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_file_size.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
