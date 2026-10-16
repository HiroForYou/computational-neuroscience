"""Export the open Luke finger-tapping dataset for the interactive fNIRS demo.

Dataset: Luke, R. & McAlpine, D. (2021). fNIRS Finger Tapping Data in BIDS Format.
Zenodo, doi:10.5281/zenodo.5529797 (open access; cite when used). Downloaded through Cedalion.

Usage (from the repository root, with the extension environment):
    extensions\\fnirs-cedalion\\.venv\\Scripts\\python.exe extensions\\fnirs-cedalion\\export_luke_demo.py

Writes interactive/data/fnirs-luke-tapping.js (window.FNIRS_DATA = {...}). Takes ~30-45 minutes
because AR-IRLS is fitted for each of the 5 subjects.
"""

import json
import os
import pickle
import time
import warnings
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
from scipy import stats
from statsmodels.stats.multitest import multipletests

import cedalion
import cedalion.data
import cedalion.io
import cedalion.models.glm as glm
import cedalion.nirs
import cedalion.sigproc.frequency
import cedalion.sigproc.motion as motion
import cedalion.sigproc.quality as quality
from cedalion import units
from cedalion.math.resample import resample

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "interactive" / "data" / "fnirs-luke-tapping.js"
CACHE = Path(__file__).resolve().parent / "outputs" / "luke_cache"  # per-subject results (ignored by Git)
CONDITIONS = {"1.0": "Control", "2.0": "Tapping/Left", "3.0": "Tapping/Right"}
CONTRAST_WINDOW = (4.0, 10.0)  # 5-s taps: HbO peaks ~5-7 s after onset
FS_GLM, EPOCH = 2.0, (-2.0, 15.0)
VIEWER_SECONDS = 300


def preprocess(path):
    rec = cedalion.io.read_snirf(str(path))[0]
    rec.stim.cd.rename_events(CONDITIONS)
    rec.stim = rec.stim[rec.stim.trial_type.isin(CONDITIONS.values())].reset_index(drop=True)
    amp = rec["amp"]
    _, sci_mask = quality.sci(amp, 10 * units.s, 0.75)
    _, psp_mask = quality.psp(amp, 10 * units.s, 0.03)
    clean = (sci_mask & psp_mask).sum("time") / sci_mask.sizes["time"]
    od = motion.wavelet(motion.tddr(cedalion.nirs.cw.int2od(amp)))
    dpf = xr.DataArray([6, 6], dims="wavelength", coords={"wavelength": amp.wavelength})
    conc = cedalion.nirs.cw.od2conc(od, rec.geo3d, dpf, spectrum="prahl")
    conc = cedalion.sigproc.frequency.freq_filter(conc, 0 * units.Hz, 0.5 * units.Hz)
    conc = resample(conc, Fs=FS_GLM)
    long, short = cedalion.nirs.split_long_short_channels(conc, rec.geo3d, distance_threshold=15 * units.mm)
    return rec, conc, long, short, clean


def contrast(design, stim, condition, sign=1.0, weights=None):
    time_ = design.common.time.values
    mask = np.zeros(time_.size)
    for _, row in stim[stim.trial_type == condition].iterrows():
        mask[(time_ >= row.onset + CONTRAST_WINDOW[0]) & (time_ <= row.onset + CONTRAST_WINDOW[1])] = 1.0
    weights = np.zeros(design.common.sizes["regressor"]) if weights is None else weights
    for i, name in enumerate(design.common.regressor.values):
        if name.startswith(f"HRF {condition} "):
            weights[i] += sign * np.trapezoid(design.common[:, i, 0] * mask, time_)
    return weights


def analyse(path):
    rec, conc, long, short, clean = preprocess(path)
    kernels = glm.basis_functions.GaussianKernels(t_pre=2 * units.s, t_post=15 * units.s,
                                                  t_delta=1.5 * units.s, t_std=2 * units.s)
    design = (glm.design_matrix.hrf_regressors(long, rec.stim, kernels)
              # 0.01 Hz: drifts slower than 100 s. With 40-50 min recordings, 0.02 Hz would add ~120 regressors.
              & glm.design_matrix.drift_cosine_regressors(conc, fmax=0.01 * units.Hz)
              & glm.design_matrix.average_short_channel_regressor(short))
    results = glm.fit(long, design, noise_model="ar_irls", ar_order=8, max_jobs=-1)

    hypotheses = {
        "Left": contrast(design, rec.stim, "Tapping/Left"),
        "Right": contrast(design, rec.stim, "Tapping/Right"),
        "Control": contrast(design, rec.stim, "Control"),
        "LeftMinusControl": contrast(design, rec.stim, "Control", -1.0, contrast(design, rec.stim, "Tapping/Left")),
        "RightMinusControl": contrast(design, rec.stim, "Control", -1.0, contrast(design, rec.stim, "Tapping/Right")),
    }
    tests = results.sm.t_test(list(hypotheses.values()))
    channels = [str(c) for c in long.channel.values]
    assert [str(c) for c in tests.channel.values] == channels
    stats_out = {}
    for h, name in enumerate(hypotheses):
        for chromo in ("HbO", "HbR"):
            p = tests.sm.p_values().sel(chromo=chromo, hypothesis=h).values
            effect = np.array([np.atleast_1d(r.effect)[h] for r in tests.sel(chromo=chromo).values])
            stats_out[f"{name}_{chromo}"] = {"p": p.tolist(), "effect": effect.tolist()}

    extract = glm.design_matrix.hrf_extract_regressors(long, rec.stim, kernels)
    hrf = {}
    for condition in CONDITIONS.values():
        mean, std = glm.predict_with_uncertainty(long, results, extract,
                                                 results.sm.params.regressor.str.startswith(f"HRF {condition}"))
        key = condition.split("/")[-1]
        for chromo in ("HbO", "HbR"):
            hrf[f"{key}_{chromo}"] = {
                "time": mean.time.values.round(2).tolist(),
                "mean": mean.sel(chromo=chromo).transpose("channel", "time").values,
                "std": std.sel(chromo=chromo).transpose("channel", "time").values,
            }

    data = long.pint.dequantify()
    relative = np.arange(EPOCH[0], EPOCH[1] + 1e-9, 1 / FS_GLM)
    blocks = {}
    for condition in CONDITIONS.values():
        onsets = rec.stim[rec.stim.trial_type == condition].onset.values
        key = condition.split("/")[-1]
        for chromo in ("HbO", "HbR"):
            series = data.sel(chromo=chromo).transpose("channel", "time")
            trials = []
            for onset in onsets:
                if onset + EPOCH[0] < float(data.time[0]) or onset + EPOCH[1] > float(data.time[-1]):
                    continue
                segment = np.array([np.interp(onset + relative, data.time.values, row) for row in series.values])
                trials.append(segment - segment[:, relative < 0].mean(axis=1, keepdims=True))
            trials = np.array(trials)
            blocks[f"{key}_{chromo}"] = {"mean": trials.mean(0), "sem": trials.std(0, ddof=1) / np.sqrt(len(trials)),
                                         "n": int(len(trials))}

    # CapTrak coordinates (metres): x from LPA to RPA, y towards the nasion. Project to a top view in mm.
    geometry = rec.geo3d.pint.to("mm").pint.dequantify()
    lpa, rpa, nasion = (geometry.sel(label=l).values for l in ("LPA", "RPA", "NASION"))
    center, lr = (lpa + rpa) / 2, (rpa - lpa) / np.linalg.norm(rpa - lpa)
    anterior = nasion - center
    anterior = anterior - (anterior @ lr) * lr
    anterior /= np.linalg.norm(anterior)
    labels = [str(l) for l in geometry.label.values if str(l)[0] in "SD" and str(l)[1:].isdigit()]
    plane = lambda xyz: [[round(float((p - center) @ lr), 1), round(float((p - center) @ anterior), 1)] for p in np.atleast_2d(xyz)]
    mid = [(geometry.sel(label=s).values + geometry.sel(label=d).values) / 2
           for s, d in zip(long.source.values, long.detector.values)]
    side = [float((m - center) @ lr) for m in mid]

    start = float(rec.stim.onset.min()) - 10
    view = data.sel(time=slice(start, start + VIEWER_SECONDS))
    viewer = {chromo: np.round(view.sel(chromo=chromo).transpose("channel", "time").values * 1000).astype(int).tolist()
              for chromo in ("HbO", "HbR")}

    return {
        "channels": channels,
        "position": plane(np.array(mid)),
        "hemisphere": ["right" if s > 0 else "left" for s in side],
        "optodes": {label: plane(geometry.sel(label=label).values)[0] for label in labels},
        "longPairs": [[str(s), str(d)] for s, d in zip(long.source.values, long.detector.values)],
        "shortPairs": [[str(s), str(d)] for s, d in zip(short.source.values, short.detector.values)],
        # Fraction of 10-s windows with SCI > 0.75 and PSP > 0.03, per long channel (shown, not used to drop channels).
        "cleanFraction": [round(float(clean.sel(channel=c)), 3) for c in channels],
        "trials": {k.split("/")[-1]: int((rec.stim.trial_type == k).sum()) for k in CONDITIONS.values()},
        "stats": stats_out,
        "hrf": hrf,
        "blocks": blocks,
        "relativeTime": relative.round(2).tolist(),
        "viewer": {"t0": round(float(view.time[0]), 2), "fs": FS_GLM, **viewer},
        "stim": [[round(float(r.onset), 2), r.trial_type.split("/")[-1]] for _, r in rec.stim.iterrows()
                 if start <= r.onset <= start + VIEWER_SECONDS],
    }


def to_nanomolar(array):
    return np.round(np.asarray(array) * 1000).astype(int).tolist()


def main():
    started = time.time()
    root = Path(cedalion.data.get_multisubject_fingertapping_path())
    subjects = {}
    CACHE.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob("sub-*/nirs/*_nirs.snirf")):
        name = path.name.split("_")[0]
        cached = CACHE / f"{name}.pkl"
        if cached.exists():
            subjects[name] = pickle.loads(cached.read_bytes())
            print(f"{name}: loaded from cache", flush=True)
            continue
        t = time.time()
        subjects[name] = analyse(path)
        cached.write_bytes(pickle.dumps(subjects[name]))
        print(f"{name}: done in {time.time() - t:.0f} s", flush=True)

    reference = next(iter(subjects.values()))["channels"]
    assert all(s["channels"] == reference for s in subjects.values()), "montages differ between subjects"

    group = {"channels": reference,
             "position": np.mean([s["position"] for s in subjects.values()], axis=0).round(1).tolist(),
             "hemisphere": next(iter(subjects.values()))["hemisphere"], "stats": {}, "hrf": {}}
    for key in next(iter(subjects.values()))["stats"]:
        effects = np.array([s["stats"][key]["effect"] for s in subjects.values()])  # subjects x channels
        t_values, p_values = stats.ttest_1samp(effects, 0.0, axis=0)
        group["stats"][key] = {"p": p_values.tolist(), "effect": effects.mean(0).tolist()}
    for key in next(iter(subjects.values()))["hrf"]:
        means = np.array([s["hrf"][key]["mean"] for s in subjects.values()])
        group["hrf"][key] = {"time": next(iter(subjects.values()))["hrf"][key]["time"],
                             "mean": means.mean(0), "std": means.std(0, ddof=1) / np.sqrt(len(means))}

    for entry in [*subjects.values(), group]:
        for key, value in entry["hrf"].items():
            value["mean"], value["std"] = to_nanomolar(value["mean"]), to_nanomolar(value["std"])
        for key, value in entry.get("blocks", {}).items():
            value["mean"], value["sem"] = to_nanomolar(value["mean"]), to_nanomolar(value["sem"])
        for value in entry["stats"].values():
            value["p"] = [float(f"{p:.4g}") for p in value["p"]]
            value["effect"] = [round(e, 3) for e in value["effect"]]

    payload = {
        "source": {
            "title": "fNIRS Finger Tapping Data in BIDS Format",
            "authors": "Robert Luke, David McAlpine",
            "doi": "10.5281/zenodo.5529797",
            "url": "https://github.com/rob-luke/BIDS-NIRS-Tapping",
            "license": "Open access (Zenodo, other-open); please cite the DOI.",
            "processedWith": f"Cedalion {cedalion.__version__}",
        },
        "units": "Concentrations in nM (integers); effects in uM*s over the contrast window.",
        "contrastWindow": CONTRAST_WINDOW,
        "subjects": subjects,
        "group": group,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    OUTPUT.write_text(f"// Generated by extensions/fnirs-cedalion/export_luke_demo.py. Do not edit.\n"
                      f"window.FNIRS_DATA = {text};\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.relative_to(ROOT).as_posix()} ({OUTPUT.stat().st_size / 1024:.0f} KB) "
          f"in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
