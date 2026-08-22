from __future__ import annotations

import subprocess
import sys


def test_public_inference_cli_help_runs_directly():
    result = subprocess.run(
        [sys.executable, "scripts/23_run_public_spar_inference.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
