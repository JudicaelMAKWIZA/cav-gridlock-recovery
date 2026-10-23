"""Exécute exclusivement les cellules 01 et 02 ; aucun épisode MetaDrive."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


PATH = (Path(__file__).resolve().parents[1]
        / "notebooks/11_ENTRAINEMENT_GRAPH_MARL_INTERACTIF.ipynb")


def main():
    notebook = nbformat.read(PATH, as_version=4)
    assert len(notebook.cells) == 33
    assert notebook.cells[2].cell_type == "code"
    assert notebook.cells[4].cell_type == "code"
    prefix = nbformat.v4.new_notebook(
        cells=[notebook.cells[index] for index in range(5)],
        metadata=notebook.metadata,
    )
    NotebookClient(prefix, timeout=90, kernel_name="cgr-metadrive-d",
                   resources={"metadata": {"path": str(PATH.parent)}}).execute()
    for index in (2, 4):
        notebook.cells[index].outputs = prefix.cells[index].outputs
        notebook.cells[index].execution_count = prefix.cells[index].execution_count
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and index not in (2, 4):
            assert not cell.outputs and cell.execution_count is None
    nbformat.write(notebook, PATH)
    print("Cellules 01 et 02 exécutées ; cellules 03–16 intactes.")


if __name__ == "__main__":
    main()
