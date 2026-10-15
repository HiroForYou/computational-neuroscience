# %% [markdown]
# # Extensión — fNIRS con Cedalion: del registro crudo a la respuesta hemodinámica
#
# **Especialización opcional (semana 17, opción “datos”).** Hasta ahora la ruta trabajó con
# modelos y datos sintéticos. Aquí se analiza un **registro real** de espectroscopía funcional
# en el infrarrojo cercano (fNIRS) con [Cedalion](https://github.com/ibs-lab/cedalion), siguiendo
# los tutoriales 3 y 4 de Middell et al. (2026, *Neurophotonics* 13(S3), S32602).
#
# **Pregunta:** ¿mover los dedos de una mano aumenta la oxigenación de la corteza motora del
# hemisferio **opuesto**?
#
# | Paso | Semana de la ruta con la que conecta |
# |---|---|
# | Calidad de señal y artefactos | 16 (procesamiento de señales) |
# | Ley de Beer–Lambert modificada | 2 (de una medida física a una variable del modelo) |
# | Filtrado y espectro | 16 |
# | GLM con respuesta hemodinámica | 12–13 (ajuste de modelos, GLM) |
# | Contrastes y corrección por comparaciones múltiples | 11 (hipótesis previas) |
#
# **Entorno:** este notebook necesita el entorno aislado de `extensions/fnirs-cedalion/`
# (ver su README). La primera ejecución descarga el dataset de ejemplo (~67 MB) a la caché de Cedalion.

# %% [markdown]
# ## 1. Qué mide fNIRS
#
# Se emite luz infrarroja (760 y 850 nm) sobre el cuero cabelludo y se mide la que vuelve a unos
# centímetros. Cada par fuente–detector es un **canal**. La hemoglobina oxigenada (HbO) y la
# desoxigenada (HbR) absorben distinto en cada longitud de onda; si cambia su concentración, cambia
# la intensidad detectada $I$. La **densidad óptica** es
#
# $$\Delta OD_\lambda(t) = -\log\frac{I_\lambda(t)}{\bar I_\lambda}$$
#
# y la **ley de Beer–Lambert modificada** la relaciona con los cambios de concentración:
#
# $$\Delta OD_\lambda = \left(\varepsilon_{\lambda}^{HbO}\,\Delta[HbO] + \varepsilon_{\lambda}^{HbR}\,\Delta[HbR]\right) \cdot d \cdot DPF_\lambda$$
#
# donde $\varepsilon$ son coeficientes de extinción, $d$ la distancia fuente–detector y $DPF$ el
# factor de longitud de camino diferencial (la luz no viaja en línea recta). Con dos longitudes de
# onda se resuelve un sistema 2×2 para $\Delta[HbO]$ y $\Delta[HbR]$.
#
# fNIRS **no mide actividad neuronal directa**: mide la respuesta hemodinámica que la acompaña,
# lenta (pico a los 5–8 s) y mezclada con fisiología sistémica (latido, respiración, ondas de Mayer).

# %%
import os
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from statsmodels.stats.multitest import multipletests

import cedalion
import cedalion.data
import cedalion.models.glm as glm
import cedalion.nirs
import cedalion.sigproc.frequency
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
OUT = ROOT / "extensions" / "fnirs-cedalion" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def save(figure, name):
    figure.tight_layout()
    figure.savefig(OUT / name, dpi=130)
    print(f"Saved: {(OUT / name).relative_to(ROOT).as_posix()}")


print("Cedalion", cedalion.__version__)
rec = cedalion.data.get_fingertappingDOT()
rec.stim.cd.rename_events({
    "1": "Control", "2": "FTapping/Left", "3": "FTapping/Right",
    "4": "BallSqueezing/Left", "5": "BallSqueezing/Right",
})
amp = rec["amp"]
fs = 1 / float(amp.time.diff("time").mean())
print(f"{amp.sizes['channel']} canales x {amp.sizes['wavelength']} longitudes de onda, "
      f"{float(amp.time[-1]) / 60:.1f} min a {fs:.2f} Hz")
print(rec.stim.trial_type.value_counts().to_dict())

# %% [markdown]
# ## 2. Calidad de señal
#
# Dos métricas por canal, en ventanas de 10 s:
#
# - **Scalp coupling index (SCI):** correlación entre las dos longitudes de onda en la banda cardiaca.
#   Si el optodo toca bien la piel, ambas ven el mismo pulso (SCI cercano a 1).
# - **Peak spectral power (PSP):** potencia del pico cardiaco. Si es muy baja, no llega señal fisiológica.
#
# Un canal se descarta si está “limpio” (SCI > 0,75 y PSP > 0,03) menos del 95 % del tiempo.

# %%
window = 10 * units.s
sci, sci_mask = quality.sci(amp, window, 0.75)
psp, psp_mask = quality.psp(amp, window, 0.03)
clean_mask = sci_mask & psp_mask
clean_fraction = clean_mask.sum("time") / clean_mask.sizes["time"]
bad_channels = list(clean_fraction.channel.values[(clean_fraction < 0.95).values])

figure, axes = plt.subplots(1, 2, figsize=(12, 3.8))
axes[0].hist(sci.values.ravel(), bins=50, color="tab:blue", alpha=0.75)
axes[0].axvline(0.75, color="k", linestyle="--", label="threshold 0.75")
axes[0].set(xlabel="SCI (per channel and 10-s window)", ylabel="count", title="Scalp coupling index")
axes[0].legend()
order = np.argsort(clean_fraction.values)
axes[1].bar(np.arange(order.size), clean_fraction.values[order] * 100,
            color=["tab:red" if v < 0.95 else "tab:green" for v in clean_fraction.values[order]])
axes[1].axhline(95, color="k", linestyle="--")
axes[1].set(xlabel="channel (sorted)", ylabel="% clean time", ylim=(50, 101), title=f"{len(bad_channels)} channels below 95 %")
save(figure, "quality.png")
print("Canales descartados:", bad_channels)

# %% [markdown]
# ## 3. Artefactos de movimiento
#
# Los movimientos de cabeza desplazan los optodos y producen saltos y picos. Se corrigen en densidad
# óptica con dos métodos encadenados:
#
# - **TDDR** (*temporal derivative distribution repair*): repondera la derivada temporal para
#   reducir los cambios bruscos que son atípicos respecto a su distribución.
# - **Wavelet:** elimina coeficientes atípicos de la descomposición wavelet (picos cortos).
#
# **Validación:** la *global variance of the temporal derivative* (GVTD), que crece con los
# artefactos, debe bajar tras la corrección.

# %%
od = cedalion.nirs.cw.int2od(amp)
od_corrected = motion.wavelet(motion.tddr(od))
gvtd_before, _ = quality.gvtd(amp)
gvtd_after, _ = quality.gvtd(cedalion.nirs.cw.od2int(od_corrected, amp.mean("time")))

figure, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(gvtd_before.time, gvtd_before, lw=0.8, label="before correction")
ax.plot(gvtd_after.time, gvtd_after, lw=0.8, label="after TDDR + wavelet")
ax.set(xlabel="time (s)", ylabel="GVTD", title="Global variance of the temporal derivative")
ax.legend()
save(figure, "motion_gvtd.png")

p95_before, p95_after = float(gvtd_before.quantile(0.95)), float(gvtd_after.quantile(0.95))
assert p95_after < p95_before, "motion correction should reduce GVTD"
print(f"GVTD percentil 95: {p95_before:.4f} -> {p95_after:.4f} ({1 - p95_after / p95_before:.0%} menos)")

# %% [markdown]
# ## 4. De densidad óptica a hemoglobina, y su espectro
#
# Se aplica Beer–Lambert modificada con $DPF = 6$ y el espectro de extinción de Prahl. El espectro
# de potencia muestra lo que el GLM tendrá que separar de la respuesta al estímulo:
#
# - **~1 Hz:** latido cardiaco.
# - **~0,1 Hz:** ondas de Mayer (oscilaciones de presión arterial), justo en la banda de la respuesta hemodinámica.
#
# Un filtro pasa-bajos a 0,5 Hz elimina el latido; las ondas de Mayer no se pueden filtrar sin
# perder la señal, por eso se modelan con **canales cortos** (sección 5).

# %%
od_clean = od_corrected.sel(channel=~od_corrected.channel.isin(bad_channels))
dpf = xr.DataArray([6, 6], dims="wavelength", coords={"wavelength": amp.wavelength})
conc = cedalion.nirs.cw.od2conc(od_clean, rec.geo3d, dpf, spectrum="prahl")
conc_filtered = cedalion.sigproc.frequency.freq_filter(conc, 0 * units.Hz, 0.5 * units.Hz)

hbo = conc.sel(chromo="HbO").pint.dequantify().values
frequencies = np.fft.rfftfreq(hbo.shape[-1], d=1 / fs)
power = (np.abs(np.fft.rfft(hbo - hbo.mean(-1, keepdims=True), axis=-1)) ** 2).mean(0)
band = lambda lo, hi: (frequencies >= lo) & (frequencies <= hi)
cardiac_peak = frequencies[band(0.7, 2.0)][np.argmax(power[band(0.7, 2.0)])]

figure, ax = plt.subplots(figsize=(10, 3.8))
ax.semilogy(frequencies, power, lw=0.8)
ax.axvline(cardiac_peak, color="tab:red", linestyle=":", label=f"cardiac peak {cardiac_peak:.2f} Hz")
ax.axvspan(0.08, 0.12, color="tab:orange", alpha=0.2, label="Mayer waves ~0.1 Hz")
ax.axvline(0.5, color="k", linestyle="--", label="low-pass cutoff 0.5 Hz")
ax.set(xlim=(0, 2.2), xlabel="frequency (Hz)", ylabel="mean HbO power", title="Power spectrum of HbO (all channels)")
ax.legend()
save(figure, "spectrum.png")

assert 0.8 < cardiac_peak < 1.8, cardiac_peak
print(f"Pico cardiaco en {cardiac_peak:.2f} Hz ({cardiac_peak * 60:.0f} latidos/min); unidades: {conc.pint.units}")

# %% [markdown]
# ## 5. GLM: separar la respuesta al estímulo de todo lo demás
#
# Para cada canal $c$ y cromóforo, el GLM explica la serie como suma de regresores:
#
# $$y_c(t) = \underbrace{\sum_{k,j} \beta^{HRF}_{k,j}\, (g_j * s_k)(t)}_{\text{respuesta a cada condición } k}
#  + \underbrace{\sum_m \beta^{drift}_m \cos(\pi m t / T)}_{\text{deriva lenta}}
#  + \underbrace{\beta^{short}\, \bar y_{short}(t)}_{\text{fisiología superficial}} + \varepsilon_c(t)$$
#
# - $s_k$ es el tren de estímulos de la condición $k$ y $g_j$ son **núcleos gaussianos** espaciados
#   1,5 s (de −2 a 15 s): la forma de la respuesta no se impone, se estima.
# - Los **canales cortos** (< 22,5 mm) solo ven cuero cabelludo; su promedio captura la fisiología
#   sistémica (incluidas las ondas de Mayer) que contamina los canales largos.
# - El ruido fNIRS está autocorrelacionado, lo que infla los $t$ de mínimos cuadrados ordinarios.
#   **AR-IRLS** ajusta un modelo autorregresivo al residuo, preblanquea y repondera de forma robusta.
#
# Se usa el registro completo (16 ensayos por mano). El ajuste tarda varios minutos.

# %%
long_channels, short_channels = cedalion.nirs.split_long_short_channels(
    conc_filtered, rec.geo3d, distance_threshold=22.5 * units.mm
)
kernels = glm.basis_functions.GaussianKernels(
    t_pre=2 * units.s, t_post=15 * units.s, t_delta=1.5 * units.s, t_std=2 * units.s
)
design = (
    glm.design_matrix.hrf_regressors(long_channels, rec.stim, kernels)
    & glm.design_matrix.drift_cosine_regressors(conc_filtered, fmax=0.02 * units.Hz)
    & glm.design_matrix.average_short_channel_regressor(short_channels)
)
results = glm.fit(long_channels, design, noise_model="ar_irls", max_jobs=-1)
print(f"{long_channels.sizes['channel']} canales largos, {short_channels.sizes['channel']} cortos; "
      f"{design.common.sizes['regressor']} regresores comunes")

# %% [markdown]
# ## 6. Hipótesis y contraste
#
# **Hipótesis (fijada antes de mirar resultados):** la actividad media entre 5 y 10 s tras el
# inicio del *tapping* es mayor que tras el control, en HbO; y el efecto aparece sobre todo en el
# hemisferio **contralateral** a la mano.
#
# El contraste integra cada regresor gaussiano en esa ventana (como en el tutorial 4 de Cedalion) y
# se prueba con un test $t$ por canal. Con 40+ canales, se corrige por comparaciones múltiples con
# **Benjamini–Hochberg** (tasa de falsos descubrimientos del 5 %).
#
# El hemisferio de cada canal se obtiene proyectando su punto medio sobre el eje que une los puntos
# preauriculares izquierdo (LPA) y derecho (RPA).

# %%
def window_contrast(condition, baseline, t_start=5.0, t_end=10.0):
    time = design.common.time.values
    masks = {condition: np.zeros(time.size), baseline: np.zeros(time.size)}
    for _, row in rec.stim.iterrows():
        if row.trial_type in masks:
            masks[row.trial_type][(time >= row.onset + t_start) & (time <= row.onset + t_end)] = 1.0
    weights = np.zeros(design.common.sizes["regressor"])
    for i, name in enumerate(design.common.regressor.values):
        for label, sign in ((condition, 1.0), (baseline, -1.0)):
            if name.startswith(f"HRF {label} "):
                weights[i] = sign * np.trapezoid(design.common[:, i, 0] * masks[label], time)
    return weights


hypotheses = {"Left": window_contrast("FTapping/Left", "Control"),
              "Right": window_contrast("FTapping/Right", "Control")}
tests = results.sm.t_test(list(hypotheses.values()))

geometry = rec.geo3d.pint.dequantify()
lpa, rpa = geometry.sel(label="LPA").values, geometry.sel(label="RPA").values
left_right_axis = (rpa - lpa) / np.linalg.norm(rpa - lpa)
midpoints = np.array([(geometry.sel(label=s).values + geometry.sel(label=d).values) / 2
                      for s, d in zip(long_channels.source.values, long_channels.detector.values)])
side = (midpoints - (lpa + rpa) / 2) @ left_right_axis  # > 0: right hemisphere

summary = {}
for h, hand in enumerate(hypotheses):
    for chromo in ("HbO", "HbR"):
        p_values = tests.sm.p_values().sel(chromo=chromo, hypothesis=h).values
        significant, q_values, _, _ = multipletests(p_values, alpha=0.05, method="fdr_bh")
        effects = tests.sm.effects.sel(chromo=chromo, hypothesis=h).values
        summary[hand, chromo] = dict(significant=significant, effects=effects, q=q_values)
        print(f"{hand:5s} {chromo}: {significant.sum():2d} canales significativos "
              f"(derecho {np.sum(significant & (side > 0))}, izquierdo {np.sum(significant & (side < 0))})")

# %% [markdown]
# ## 7. Mapa de activación
#
# Cada punto es un canal largo en el plano de la cabeza (vista superior). Color: efecto en HbO
# (µM·s en la ventana 5–10 s). Borde negro: significativo tras la corrección.

# %%
nasion, inion = geometry.sel(label="Nz").values, geometry.sel(label="Iz").values
front_axis = nasion - inion
front_axis = front_axis - (front_axis @ left_right_axis) * left_right_axis
front_axis /= np.linalg.norm(front_axis)
x_plane = (midpoints - (lpa + rpa) / 2) @ left_right_axis
y_plane = (midpoints - (lpa + rpa) / 2) @ front_axis

figure, axes = plt.subplots(1, 2, figsize=(11, 5))
limit = max(np.abs(summary[hand, "HbO"]["effects"]).max() for hand in hypotheses)
for ax, hand in zip(axes, hypotheses):
    entry = summary[hand, "HbO"]
    scatter = ax.scatter(x_plane, y_plane, c=entry["effects"], cmap="RdBu_r", vmin=-limit, vmax=limit,
                         s=110, edgecolors=np.where(entry["significant"], "k", "none"), linewidths=2)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set(title=f"{hand}-hand tapping vs control (HbO)", xlabel="left  <-  (mm)  ->  right",
           ylabel="posterior  <-  (mm)  ->  anterior", aspect="equal")
figure.colorbar(scatter, ax=axes, shrink=0.8, label="HbO effect (window 5-10 s)")
figure.savefig(OUT / "activation_map.png", dpi=130)
print(f"Saved: {(OUT / 'activation_map.png').relative_to(ROOT).as_posix()}")

# %% [markdown]
# ## 8. Respuesta hemodinámica estimada
#
# Con los coeficientes de los núcleos gaussianos se reconstruye la forma de la respuesta (± 1 error
# estándar) en el canal más significativo de cada mano. Se espera la firma clásica: HbO sube y HbR
# baja algo menos, con pico a los 5–8 s.

# %%
extract = glm.design_matrix.hrf_extract_regressors(long_channels, rec.stim, kernels)
figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
peaks = {}
for ax, hand in zip(axes, hypotheses):
    best = int(np.argmin(summary[hand, "HbO"]["q"]))
    channel = long_channels.channel.values[best]
    mean, std = glm.predict_with_uncertainty(
        long_channels, results, extract, results.sm.params.regressor.str.startswith(f"HRF FTapping/{hand}")
    )
    for chromo, color in (("HbO", "tab:red"), ("HbR", "tab:blue")):
        m, s = mean.sel(channel=channel, chromo=chromo), std.sel(channel=channel, chromo=chromo)
        ax.plot(m.time, m, color=color, label=chromo)
        ax.fill_between(m.time, m - s, m + s, color=color, alpha=0.25)
    hbo_curve = mean.sel(channel=channel, chromo="HbO")
    peaks[hand] = (channel, float(hbo_curve.time[int(hbo_curve.argmax())]), float(hbo_curve.max()),
                   float(mean.sel(channel=channel, chromo="HbR").min()))
    ax.axvline(0, color="k", lw=1)
    ax.axvspan(0, 10, color="grey", alpha=0.1, label="tapping (10 s)")
    ax.set(title=f"{hand} hand - channel {channel} ({'right' if side[best] > 0 else 'left'} hemisphere)",
           xlabel="time from onset (s)", ylabel="concentration change (µM)")
    ax.legend()
save(figure, "hrf.png")

# %% [markdown]
# ## 9. Validación
#
# 1. Hay canales con activación significativa en HbO para cada mano.
# 2. **Lateralización:** la mayoría de esos canales están en el hemisferio contralateral a la mano.
# 3. **Signo:** en esos canales el efecto de HbO es positivo (aumento de oxigenación).
# 4. **Forma:** en el mejor canal, HbO tiene su pico entre 3 y 12 s y HbR baja (mínimo negativo).

# %%
for hand, contralateral in (("Left", 1), ("Right", -1)):
    entry = summary[hand, "HbO"]
    significant = entry["significant"]
    n_contra = int(np.sum(significant & (np.sign(side) == contralateral)))
    channel, t_peak, hbo_peak, hbr_min = peaks[hand]
    assert significant.sum() > 0, f"no significant channels for {hand}"
    assert n_contra / significant.sum() >= 0.7, (hand, n_contra, int(significant.sum()))
    assert np.all(entry["effects"][significant] > 0), f"negative HbO effect for {hand}"
    assert 3 <= t_peak <= 12 and hbo_peak > 0 and hbr_min < 0, peaks[hand]
    print(f"{hand}: {n_contra}/{int(significant.sum())} canales significativos son contralaterales; "
          f"pico HbO {hbo_peak:.3f} µM a {t_peak:.1f} s en {channel}; mínimo HbR {hbr_min:.3f} µM")

# %% [markdown]
# ## 10. Interpretación y límites
#
# - El resultado reproduce la organización contralateral de la corteza motora con una medida
#   indirecta (hemodinámica) y un modelo lineal: la misma lógica de ajuste e hipótesis de las semanas
#   11–13, ahora con datos reales.
# - **Un solo sujeto y una sesión:** no permite generalizar a la población (haría falta un análisis
#   de grupo, por ejemplo con `get_multisubject_fingertapping_path`).
# - **Decisiones del análisis:** umbrales de calidad, métodos de corrección, DPF, filtro y ventana
#   del contraste afectan el resultado. El estudio FRESH (citado en el artículo) mostró que equipos
#   distintos llegan a conclusiones distintas con los mismos datos: documenta y justifica cada elección.
# - **Siguiente paso:** el tutorial 5 de Cedalion reconstruye esta activación en el espacio de la
#   corteza (DOT), y el 6 la decodifica ensayo a ensayo con validación cruzada (semana 14).
#
# **Para tu entrega:** cambia la ventana del contraste a 10–15 s o quita el regresor de canales
# cortos, predice el efecto antes de ejecutar y compara el número de canales significativos.

# %% [markdown]
# ## Referencias de los métodos usados
#
# - Middell, E. et al. (2026). Cedalion tutorial: a Python-based framework for comprehensive analysis
#   of multimodal fNIRS and DOT. *Neurophotonics* 13(S3), S32602. doi:10.1117/1.NPh.13.S3.S32602
# - Pollonini, L. et al. (2016). PHOEBE: a method for real time mapping of optodes-scalp coupling in
#   functional near-infrared spectroscopy. *Biomedical Optics Express* 7(12) — SCI y PSP.
# - Fishburn, F. A. et al. (2019). Temporal Derivative Distribution Repair (TDDR). *NeuroImage* 184.
# - Molavi, B. & Dumont, G. A. (2012). Wavelet-based motion artifact removal for fNIRS.
#   *Physiological Measurement* 33(2).
# - Sherafati, A. et al. (2020). Global motion detection and censoring in high-density diffuse optical
#   tomography (GVTD). *Human Brain Mapping* 41(14).
# - Barker, J. W., Aarabi, A. & Huppert, T. J. (2013). Autoregressive model based algorithm for
#   correcting motion and serially correlated errors in fNIRS (AR-IRLS). *Biomedical Optics Express* 4(8).
