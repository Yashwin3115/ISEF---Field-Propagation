"""Part 3 (steps 14-17): compare MaxOne array recordings with a COMMITTED
prediction. Runs only when the lab supplies recordings; nothing here may be
used to change a prediction.

Input: a per-electrode CSV made from the recording (the raw MaxWell file
format is not known yet -- the conversion step is written once it is):
    col, row, area_Vs, clipped
  area_Vs   integral of the artifact over the FIRST phase of the pulse,
            baseline subtracted, averaged over repeated pulses (V x s).
            Because the artifact is Z_k x i(t), this area is Z_k x Q (the
            charge of the phase) -- independent of the amplifier's bandwidth
            and of exactly where the 50 us samples fall.
  clipped   1 if any sample in the phase was clipped (excluded, step 17)

Two comparisons, reported separately (plan 5.13):
  BLIND     predicted area = Z_k x Q_pred, Q_pred from the committed interface
            model (charge per phase) -- no number taken from the data.
  PATTERN   one overall scale fitted (the charge), so only the SHAPE across
            the array is tested. Labelled "fitted".
For every model version: full model with each return path, point source
I/(2 pi sigma r). Relative error vs distance, coefficient of determination
across electrodes, and which return path the data prefer.

    python runs/array_compare.py results/predictions  data/array/areas.csv  [--amplitude 0.5]
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import electrode_xy, PITCH, PROVISIONAL, um
from phantom_analysis import read_csv

args = [a for a in sys.argv[1:] if not a.startswith("--")]
amp = float(sys.argv[sys.argv.index("--amplitude") + 1]) if "--amplitude" in sys.argv else 0.5
args = [a for a in args if a != str(amp)]
if len(args) < 2:
    sys.exit(__doc__)
pred_dir, csv_path = Path(args[0]), args[1]


def load_prediction(pred_dir):
    """The committed prediction: per-electrode transfer resistance for each
    return path (CSV) and the predicted charge per phase (summary.json)."""
    s = json.loads((pred_dir / "summary.json").read_text())
    csvf = next(pred_dir.glob("*predictions.csv"))
    lines = [l for l in csvf.read_text().splitlines() if not l.startswith("#")]
    hdr = lines[0].split(",")
    data = np.array([[float(v) for v in l.split(",")] for l in lines[1:]])
    cols = {h: data[:, i] for i, h in enumerate(hdr)}
    Z = {h[2:-4]: cols[h] for h in hdr if h.startswith("Z_") and h.endswith("_ohm")}
    return s, cols["col"].astype(int), cols["row"].astype(int), Z


s, pc, pr, Zs = load_prediction(pred_dir)
rows = read_csv(csv_path)
col = np.array([r["col"] for r in rows], int); row = np.array([r["row"] for r in rows], int)
area = np.array([r["area_Vs"] for r in rows]); clipped = np.array([r.get("clipped", 0) for r in rows]) > 0
p = s.get("settings", PROVISIONAL)
c0, r0 = p["stim_col_row"]
sx, sy = electrode_xy(c0, r0)
ex, ey = electrode_xy(col, row)
dist = np.hypot(ex - sx, ey - sy)
use = ~clipped & (dist > 0)
print(f"{len(rows)} electrodes in the file, {clipped.sum()} clipped (excluded), {use.sum()} used")

report = {"amplitude_V": amp, "n_used": int(use.sum()), "prediction": str(pred_dir), "models": {}}
models, Qrange = {}, {}
lut_idx = {(a, b): i for i, (a, b) in enumerate(zip(pc, pr))}
ix = np.array([lut_idx.get((a, b), -1) for a, b in zip(col, row)])
for ret, Z in Zs.items():
    models[f"full model, {ret}"] = np.where(ix >= 0, Z[np.maximum(ix, 0)], np.nan)
    pulses = s["interface"]["by_return"].get(ret, {}).get("pulses", [])
    qs = [q["charge_phase1_C"] for q in pulses if abs(q["amplitude_V"] - amp) < 1e-9]
    Qrange[f"full model, {ret}"] = (min(qs), max(qs)) if qs else None
sigma = p.get("sigma_S_per_m", 1.5)
models["point source"] = 1 / (2 * np.pi * sigma * np.maximum(dist, 1e-9))
Qrange["point source"] = Qrange.get("full model, top_plate")

bands = [(1, 5), (5, 20), (20, 60), (60, 130)]
fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
for name, Z in models.items():
    m = use & np.isfinite(Z)
    out = {}
    Qr = Qrange.get(name)
    if Qr is not None:
        lo, hi = Z[m] * Qr[0], Z[m] * Qr[1]                     # committed range (|Z| 300-600 kOhm)
        out["blind_fraction_inside_committed_range"] = float(np.mean((area[m] >= lo) & (area[m] <= hi)))
        out["blind_median_abs_rel_error_vs_mid"] = float(np.median(np.abs(area[m] / (0.5 * (lo + hi)) - 1)))
    scale = np.sum(area[m] * Z[m]) / np.sum(Z[m] ** 2)          # fitted charge
    fit = scale * Z[m]
    relf = area[m] / fit - 1
    ss_res = np.sum((area[m] - fit) ** 2); ss_tot = np.sum((area[m] - area[m].mean()) ** 2)
    out["fitted_charge_C"] = float(scale)
    out["pattern_R2"] = float(1 - ss_res / ss_tot)
    out["pattern_median_abs_rel_error"] = float(np.median(np.abs(relf)))
    dp = dist[m] / PITCH
    out["pattern_rel_error_by_band"] = {f"{a}-{b} pitches": float(np.median(np.abs(relf[(dp >= a) & (dp < b)])))
                                        for a, b in bands if np.any((dp >= a) & (dp < b))}
    report["models"][name] = out
    ax.semilogx(dp, relf * 100, ".", ms=2, alpha=0.5, label=f"{name} (R^2 {out['pattern_R2']:.3f})")
    print(f"{name:28s} pattern R^2 {out['pattern_R2']:.4f}, median |error| "
          f"{out['pattern_median_abs_rel_error']*100:.1f}%"
          + (f", blind: {out['blind_fraction_inside_committed_range']*100:.0f}% inside the committed range"
             if "blind_fraction_inside_committed_range" in out else ""))
best = max((k for k in report["models"] if k.startswith("full")), key=lambda k: report["models"][k]["pattern_R2"])
report["preferred_return_path"] = best
print(f"data prefer: {best}")
ax.axhline(0, color="0.5", lw=1)
ax.set_xlabel("distance from the stimulating pad (pitches)")
ax.set_ylabel("(measured - fitted pattern) / fitted (%)")
ax.legend(frameon=False, fontsize=8, markerscale=4)
out_dir = ROOT / "results" / "array_comparison" / pred_dir.name
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "pattern_residuals.png", dpi=180)
(out_dir / "comparison.json").write_text(json.dumps(report, indent=2))
print(f"wrote {out_dir.relative_to(ROOT)}/")
