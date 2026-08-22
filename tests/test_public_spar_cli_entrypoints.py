from __future__ import annotations

import subprocess
import sys
import importlib.util
from pathlib import Path


def test_public_spar_cli_entrypoints_support_direct_script_execution():
    for script in (
        "scripts/19_inspect_public_spar.py",
        "scripts/20_prepare_public_spar.py",
        "scripts/21_build_public_spar_geometry.py",
        "scripts/22_build_public_spar_prompts.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (script, result.stdout, result.stderr)


def test_public_spar_dataset_entrypoints_default_to_the_available_test_split(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    for filename in ("19_inspect_public_spar.py", "20_prepare_public_spar.py", "21_build_public_spar_geometry.py"):
        path = repo_root / "scripts" / filename
        spec = importlib.util.spec_from_file_location(filename.replace(".", "_"), path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        extra = []
        if filename.startswith("20"):
            extra = ["--phase", "a"]
        elif filename.startswith("21"):
            extra = ["--subset", "subset.json"]
        monkeypatch.setattr(sys, "argv", [filename] + extra)
        assert module.parse_args().split == "test"
