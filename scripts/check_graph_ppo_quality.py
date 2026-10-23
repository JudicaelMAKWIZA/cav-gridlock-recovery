"""Exécute les contrôles de non-régression avant de figer la campagne."""

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time

import nbformat

from cav_recovery.laboratory.storage import atomic_json, file_hash


PROJECT = Path(__file__).resolve().parents[1]


def command(label, arguments, output):
    started = time.perf_counter()
    process = subprocess.run(arguments, cwd=PROJECT, text=True,
                             capture_output=True, check=False)
    log = output / f"{label}.log"
    log.write_text(process.stdout + process.stderr, encoding="utf-8")
    combined = process.stdout + process.stderr
    native_fault = any(marker in combined for marker in (
        "Windows fatal exception", "Fatal Python error", "access violation",
    ))
    return dict(exit_code=process.returncode, native_fault=native_fault,
                wall_s=time.perf_counter() - started,
                log=str(log), log_sha256=file_hash(log),
                tail=combined[-1200:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPYCACHEPREFIX"] = "D:/cav-recovery-cache/graph-ppo-pycache"
    os.environ["TEMP"] = "D:/cav-recovery-temp"
    os.environ["TMP"] = "D:/cav-recovery-temp"
    os.environ["PYTHONPATH"] = str(PROJECT / "src")
    os.environ["MPLBACKEND"] = "Agg"
    checks = {
        "compile": command("compile", [sys.executable, "-m", "compileall", "-q",
                                       "src/cav_recovery", "tests", "scripts"], output),
        "unit": command("unit", [sys.executable, "-m", "pytest", "tests/unit", "-q"],
                        output),
        "metadrive": command("metadrive", [
            sys.executable, "-c",
            "import torch; import pytest; "
            "import sys; sys.exit(pytest.main(sys.argv[1:]))",
            "tests/integration/test_metadrive.py",
            "tests/integration/test_recovery_native.py",
            "tests/integration/test_recovery_actuation_v2.py",
            "tests/integration/test_recovery_budget_v3.py",
            "-q",
        ], output),
        "dependencies": command("dependencies", [sys.executable, "-m", "pip", "check"],
                                output),
        "git_diff": command("git_diff", ["git", "diff", "--check"], output),
    }
    notebook = PROJECT / "notebooks/12_ENTRAINEMENT_GRAPH_PPO_COMPLET.ipynb"
    content = nbformat.read(notebook, as_version=4)
    nbformat.validate(content)
    for cell in content.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    checks["notebook"] = dict(exit_code=0, code_cells=sum(
        cell.cell_type == "code" for cell in content.cells),
        sha256=file_hash(notebook))
    atomic_json(output / "quality-checks.json", dict(
        interpreter=sys.executable, checks=checks,
        tools_available={name: importlib.util.find_spec(name) is not None
                         for name in ("sourcery", "ruff", "radon", "pylint", "mypy")},
        passed=all(check["exit_code"] == 0 and not check.get("native_fault")
                   for check in checks.values()),
    ))
    print({name: (check["exit_code"], check.get("native_fault", False))
           for name, check in checks.items()})
    return 0 if all(check["exit_code"] == 0 and not check.get("native_fault")
                    for check in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
