"""Run local notebook code cells without requiring a Jupyter service."""
import os
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        namespace = {"__name__": "__main__"}
        count = 0
        for cell in notebook.cells:
            if cell.cell_type == "code":
                exec(compile(cell.source, str(path), "exec"), namespace)
                count += 1
        print(f"{path.name}: {count} code cells executed")


if __name__ == "__main__":
    main()
