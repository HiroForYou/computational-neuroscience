"""Convierte un notebook editado en Jupyter de vuelta a su script con celdas `# %%`.

Por defecto escribe el .py con el mismo nombre en la misma carpeta, que en weeks/week_XX_<tema>/ es
el script fuente: no se crean copias. Las salidas del notebook se descartan.

Uso:
    python tools/convert_from_notebooks.py weeks/week_03_lif_currents/week_03_lif_currents.ipynb
    python tools/convert_from_notebooks.py otro.ipynb destino.py
"""

import json
import sys
from pathlib import Path


def _source_to_lines(source):
    if source is None:
        return []
    if isinstance(source, str):
        return source.splitlines(keepends=True)
    return list(source)


def convertir_ipynb_a_py(archivo_ipynb, archivo_py=None):
    archivo_ipynb = Path(archivo_ipynb)

    if not archivo_ipynb.exists():
        print(f"Error: No existe el archivo {archivo_ipynb}")
        return

    with open(archivo_ipynb, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    archivo_py = Path(archivo_py) if archivo_py else archivo_ipynb.with_suffix(".py")
    bloques = []
    for celda in notebook.get("cells", []):
        cell_type = celda.get("cell_type")
        source = _source_to_lines(celda.get("source", []))

        if cell_type == "markdown":
            lineas = ["# %% [markdown]"]
            for linea in source:
                contenido = linea.rstrip("\n").rstrip()
                lineas.append("# " + contenido if contenido else "#")
            bloques.append("\n".join(lineas))

        elif cell_type == "code":
            codigo = "".join(source).rstrip("\n")
            bloques.append("# %%\n" + codigo)

    with open(archivo_py, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n\n".join(bloques) + "\n")

    print(f"Convertido correctamente: {archivo_py}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Uso: python tools/convert_from_notebooks.py archivo.ipynb [destino.py]")
    else:
        convertir_ipynb_a_py(*sys.argv[1:])
