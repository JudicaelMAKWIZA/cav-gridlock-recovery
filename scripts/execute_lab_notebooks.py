"""Exécute les cahiers publics vers une copie privée, avec un noyau sur D:."""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

from cav_recovery.laboratory.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    kernel_root = runtime / "kernels"
    atomic_json(
        kernel_root / "cav-recovery-lab" / "kernel.json",
        dict(
            argv=[
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ],
            display_name="CAV Recovery Windows CPU",
            language="python",
        ),
    )
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime / "connections")
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(exist_ok=True)
    checks = []
    for source in sorted((repo / "notebooks").glob("0*.ipynb")):
        notebook = nbformat.read(source, as_version=4)
        manager = KernelManager(
            kernel_name="cav-recovery-lab",
            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
            connection_file=str(runtime / "connections" / (source.stem + ".json")),
        )
        client = NotebookClient(
            notebook,
            km=manager,
            timeout=1200,
            resources={"metadata": {"path": str(repo)}},
        )
        print("Exécution privée : " + source.name, flush=True)
        try:
            executed = client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(executed, args.output / source.name)
        errors = [
            out
            for cell in executed.cells
            for out in cell.get("outputs", [])
            if out.output_type == "error"
        ]
        assert not errors
        checks.append(
            dict(
                notebook=source.name,
                executed_code_cells=sum(c.cell_type == "code" for c in executed.cells),
                errors=len(errors),
            )
        )
        atomic_json(args.output / "execution.json", checks)
    print("Cinq cahiers exécutés, aucune sortie privée écrite dans Git.", flush=True)


if __name__ == "__main__":
    main()
