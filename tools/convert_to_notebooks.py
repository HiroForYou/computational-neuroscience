"""Convierte scripts con celdas `# %%` / `# %% [markdown]` en notebooks de Jupyter.

Cada semana vive en weeks/week_XX_<tema>/: el script .py es la fuente y el .ipynb se genera
a su lado, con el mismo nombre.

Uso:
    python tools/convert_to_notebooks.py                 # todas las semanas
    python tools/convert_to_notebooks.py --execute       # todas, ejecutadas (con figuras)
    python tools/convert_to_notebooks.py weeks/week_03_lif_currents/week_03_lif_currents.py --execute
    python tools/convert_to_notebooks.py --reuse-outputs   # solo cambió texto: conserva las figuras sin ejecutar
"""

import argparse
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEKS_DIR = ROOT / "weeks"


def parse_cells(content):
    """Divide un script en celdas (tipo, nombre, líneas) según los marcadores `# %%`."""
    cells = []
    current_cell = []
    current_cell_type = None
    current_cell_name = None

    def save_current_cell():
        while current_cell and current_cell[-1].strip() == "":
            current_cell.pop()
        if current_cell and current_cell_type:
            cells.append((current_cell_type, current_cell_name, list(current_cell)))

    for line in content.split("\n"):
        if line.strip() == "# %% [markdown]":
            save_current_cell()
            current_cell, current_cell_type, current_cell_name = [], "markdown", None
            continue
        if line.strip() == "# %%":
            save_current_cell()
            current_cell, current_cell_type, current_cell_name = [], "code", None
            continue

        if current_cell_type == "markdown":
            # Quitar el '#' inicial; una línea '#' sola es una línea vacía del markdown.
            if line.startswith("# "):
                text = line[2:]
                current_cell.append(text)
                if not current_cell_name and text.startswith(("# ", "## ", "### ", "#### ")):
                    current_cell_name = text.lstrip("#").strip() or None
            elif line.strip() in ("", "#"):
                current_cell.append("")
            else:
                current_cell.append(line)
        else:
            if not current_cell_type:
                current_cell_type = "code"
            if current_cell or line.strip():
                current_cell.append(line)

    save_current_cell()
    return cells


def build_notebook(content):
    """Crea la estructura nbformat 4.5 a partir del texto de un script."""
    cells = []
    for cell_type, name, lines in parse_cells(content):
        cell = {
            "cell_type": cell_type,
            "id": uuid.uuid4().hex[:12],
            "metadata": {"name": name} if name else {},
            "source": "\n".join(lines),
        }
        if cell_type == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute_notebook(notebook_path):
    """Ejecuta el notebook en su lugar para guardar las salidas y figuras."""
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(notebook_path, as_version=4)
    NotebookClient(notebook, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, notebook_path)


def reuse_outputs(notebook, previous_path):
    """Copia las salidas de las celdas de código cuyo texto no cambió; devuelve cuántas faltan."""
    previous = json.loads(Path(previous_path).read_text(encoding="utf-8"))
    as_text = lambda source: "".join(source) if isinstance(source, list) else source
    executed = {as_text(c["source"]): c for c in previous["cells"] if c["cell_type"] == "code"}
    missing = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        old = executed.get(cell["source"])
        if old is None or old.get("execution_count") is None:
            missing += 1
            continue
        cell["outputs"], cell["execution_count"] = old["outputs"], old["execution_count"]
    return missing


def convert_py_to_notebook(py_file_path, output_notebook_path, execute=False, reuse=False):
    content = Path(py_file_path).read_text(encoding="utf-8")
    output_notebook_path = Path(output_notebook_path)
    output_notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(content)
    note = ""
    if reuse and output_notebook_path.exists():
        missing = reuse_outputs(notebook, output_notebook_path)
        note = (" (salidas reutilizadas)" if not missing
                else f" (ATENCIÓN: {missing} celda(s) de código sin salida; ejecútalo con --execute)")
    with open(output_notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
        f.write("\n")
    if execute:
        execute_notebook(output_notebook_path)
        note = " (ejecutado)"
    print(f"Convertido: {py_file_path} -> {output_notebook_path}{note}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scripts", nargs="*", type=Path, help="scripts a convertir (por defecto, todas las semanas)")
    parser.add_argument("--execute", action="store_true", help="ejecutar cada notebook para incluir figuras y salidas")
    parser.add_argument("--reuse-outputs", action="store_true",
                        help="sin ejecutar: conservar las salidas de las celdas de código que no cambiaron")
    args = parser.parse_args()

    scripts = args.scripts or sorted(WEEKS_DIR.glob("week_*/week_*.py"))
    for py_path in scripts:
        if not py_path.exists():
            print(f"WARNING: No encontrado: {py_path}")
            continue
        convert_py_to_notebook(py_path, py_path.with_suffix(".ipynb"), execute=args.execute, reuse=args.reuse_outputs)
    print(f"\nConversion completada! {len(scripts)} notebook(s) junto a sus scripts en {WEEKS_DIR}")


if __name__ == "__main__":
    main()
