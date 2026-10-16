# Extensión: fNIRS con Cedalion

Especialización opcional de la **semana 17** (opción “datos”). Analiza un registro **real** de
fNIRS (*finger tapping*) con [Cedalion](https://github.com/ibs-lab/cedalion), siguiendo los
tutoriales 3 y 4 de:

> Middell, E. et al. (2026). Cedalion tutorial: a Python-based framework for comprehensive analysis
> of multimodal fNIRS and DOT from the lab to the everyday world. *Neurophotonics* 13(S3), S32602.
> doi:[10.1117/1.NPh.13.S3.S32602](https://doi.org/10.1117/1.NPh.13.S3.S32602)

| Archivo | Contenido |
|---|---|
| `fnirs_finger_tapping.py` | Fuente (celdas `# %%`): calidad de señal, corrección de movimiento, Beer-Lambert, espectro, GLM con AR-IRLS, contraste con FDR, lateralización y validación |
| `fnirs_finger_tapping.ipynb` | Notebook generado y ejecutado, con las figuras |
| `requirements.txt` | Dependencias directas (Cedalion fijado a v26.5.1) |
| `requirements-lock.txt` | Versiones exactas de un entorno que funcionó (Windows, Python 3.11) |
| `outputs/` | Figuras generadas (ignorado por Git) |

## Por qué un entorno aparte

Cedalion fija `numpy==2.2.*` y `matplotlib==3.10.*`, exige Python ≥ 3.11 y trae unas 140
dependencias (MNE, VTK, PyVista, simuladores de fotones…). Instalarlo en el entorno principal
bajaría de versión sus paquetes. Esta extensión tiene su propio entorno y **no forma parte de la
suite de tests** del proyecto: los tests solo comprueban que el notebook esté sincronizado con su script.

## Instalación (Windows, PowerShell)

Desde la raíz del repositorio:

```powershell
py -3.11 -m venv extensions\fnirs-cedalion\.venv
extensions\fnirs-cedalion\.venv\Scripts\python.exe -m pip install -r extensions\fnirs-cedalion\requirements.txt
```

La instalación tarda ~10 minutos. Para reproducir exactamente el entorno probado, usa
`requirements-lock.txt` en lugar de `requirements.txt`.

## Ejecución

```powershell
# Regenerar y ejecutar el notebook (desde la raíz del repo, con el entorno de la extensión)
extensions\fnirs-cedalion\.venv\Scripts\python.exe tools\convert_to_notebooks.py extensions\fnirs-cedalion\fnirs_finger_tapping.py --execute
```

O abre `fnirs_finger_tapping.ipynb` en VS Code/Jupyter y elige como kernel el Python de
`extensions\fnirs-cedalion\.venv`.

- La primera ejecución descarga el dataset de ejemplo (~67 MB) a la caché de Cedalion
  (`%LOCALAPPDATA%\cedalion`), fuera del repositorio.
- El ajuste del GLM con AR-IRLS es la parte lenta (~6 minutos, ~8 en total). Por eso se analizan
  los primeros 1000 s del registro (8 ensayos por mano): con el registro completo (~34 min) el
  ajuste pasa de 30 minutos.
- Entre ejecuciones, las amplitudes pueden variar hasta ~10% (por ejemplo, el pico de HbO de
  S10D27 osciló entre 0.076 y 0.085 µM en distintas ejecuciones). Los recuentos de canales
  significativos y las conclusiones se mantuvieron idénticos.
- Si solo editas texto (celdas markdown), regenera sin volver a ejecutar:
  `python tools\convert_to_notebooks.py extensions\fnirs-cedalion\fnirs_finger_tapping.py --reuse-outputs`

## Datos y licencias

| Dataset | Uso aquí | Condiciones |
|---|---|---|
| *Finger tapping* HD-DOT del IBS Lab (`cedalion.data.get_fingertappingDOT`) | Notebook | Su README dice: *“Please use only for Cedalion example notebooks. No other use permitted.”* El repositorio no incluye los datos; el notebook los descarga al ejecutarse. |
| *fNIRS Finger Tapping Data in BIDS Format* (R. Luke y D. McAlpine, [doi:10.5281/zenodo.5529797](https://doi.org/10.5281/zenodo.5529797)) | Demo interactiva [`interactive/fnirs.html`](../../interactive/fnirs.html) | Acceso abierto en Zenodo (licencia *other-open*); citar el DOI. 5 sujetos, 30 ensayos por condición. |

Por esa restricción, la demo **no** usa los datos del IBS Lab: usa el dataset abierto de Luke.

El código del notebook adapta los tutoriales de Cedalion (licencia MIT, © 2024-2025 The Cedalion
Developers). Material educativo, sin uso clínico ni diagnóstico.

## Demo interactiva con datos reales

`export_luke_demo.py` procesa los 5 sujetos con el mismo flujo del notebook (calidad, TDDR + wavelet,
Beer-Lambert, filtro, GLM con AR-IRLS y canales cortos) y escribe
`interactive/data/fnirs-luke-tapping.js`, que la demo carga sin servidor:

```powershell
extensions\fnirs-cedalion\.venv\Scripts\python.exe extensions\fnirs-cedalion\export_luke_demo.py
```

Tarda ~18 minutos (AR-IRLS sobre registros de 40-50 min). Guarda en caché el resultado de cada sujeto en
`outputs/luke_cache/`, así que si se interrumpe continúa donde quedó; borra esa carpeta para recalcular.

## Resultado en una línea

El *tapping* con la mano izquierda activa el hemisferio contralateral (11 de 12
canales, LI = +0.83); con la mano derecha la tendencia es contralateral pero no sobrevive a la
corrección por comparaciones múltiples. El notebook explica por qué el contraste con la condición
“Control” del tutorial original lleva a una conclusión engañosa.
