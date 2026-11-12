"""Compare phantom measurements with the COMMITTED predictions (plan 5.11).

    python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 \\
        data/phantom/measurements.csv data/phantom/sessions.csv [data/phantom/reciprocity.csv]
        [--reference separate]   if the scope was referenced to a separate far wire

Templates for the CSV files: notes/phantom_*_template.csv. Plateau voltages
come from runs/scope_plateau.py (or read by hand from the scope).

Reports, in this order (the first is the headline, fixed in advance):
  1. coverage: fraction of measurements inside the committed 95% band
  2. RMSE and relative error per model version (Part 6)
  3. paired Wilcoxon signed-rank test, full model vs each idealized model
  4. decay exponent with a bootstrap 95% CI (sessions, then positions)
  5. linearity in current, reciprocity, conductivity drift
Nothing here changes a prediction. Any refit is a separate, labelled step.
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
from phantom_analysis import (read_csv, predict_rows, coverage, errors_by_model,
                              wilcoxon_vs_full, decay_exponent, reciprocity, linearity)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
ref = sys.argv[sys.argv.index("--reference") + 1] if "--reference" in sys.argv else "return"
args = [a for a in args if a != ref]
if len(args) < 3:
    sys.exit(__doc__)
pred_dir, meas_csv, sess_csv = Path(args[0]), args[1], args[2]
recip_csv = args[3] if len(args) > 3 else None
pred = json.loads((pred_dir / "predictions.json").read_text())
if pred.get("quick"):
    print("WARNING: these are QUICK predictions (coarse grids), not a committed set.")

rows = predict_rows(read_csv(meas_csv), read_csv(sess_csv), pred, reference=ref)
out_dir = ROOT / "results" / "phantom_comparison" / pred_dir.name
fig_dir = ROOT / "figures" / "phantom_comparison" / pred_dir.name
out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
report = {"predictions": str(pred_dir), "reference": ref}

for part in (6, 7):
    pr = [r for r in rows if int(r["part"]) == part]
    if not pr:
        continue
    cov, k, n = coverage(pr)
    print(f"\n=== PART {part}: {n} measurements, {len(set(str(r['session']) for r in pr))} sessions ===")
    print(f"1. COVERAGE: {k}/{n} = {cov*100:.0f}% inside the committed 95% band")
    miss = [r for r in pr if not r["covered"]]
    for r in miss:
        print(f"   outside: session {r['session']}, position {r['position']} "
              f"({r['distance_mm']:.1f} mm), {r.get('pulse_width_us')} us: measured "
              f"{r['plateau_V']*1e3:.3f} mV, predicted {r['V_pred']*1e3:.3f} +- {r['band_V']*1e3:.3f} mV")
    rep = {"coverage": cov, "covered": k, "n": n,
           "outside": [{k_: r[k_] for k_ in ("session", "position", "distance_mm", "pulse_width_us",
                                             "plateau_V", "V_pred", "band_V")} for r in miss]}
    errs = errors_by_model(pr)
    print("2. error per model version")
    for name, e in errs.items():
        print(f"   {name:32s} RMSE {e['rmse_V']*1e3:8.3f} mV   median |rel. error| "
              f"{e['median_abs_rel_error']*100:6.1f}%")
    rep["errors"] = {k_: {kk: v for kk, v in e.items() if kk != "rel_errors"} for k_, e in errs.items()}
    if part == 6:
        w = wilcoxon_vs_full(pr)
        print("3. paired Wilcoxon signed-rank (H1: full model has smaller |error|)")
        for name, v in w.items():
            if v["p"] is None:
                print(f"   vs {name:30s} n = {v['n']}: {v['note']}")
            else:
                print(f"   vs {name:30s} n = {v['n']}, full better in {v['full_model_better_in']}, "
                      f"p = {v['p']:.2g}")
        rep["wilcoxon"] = w
        dex = decay_exponent(pr)
        print(f"4. decay exponent (V ~ r^-n, all distances): n = {dex['exponent']:.3f}, "
              f"95% CI [{dex['ci95'][0]:.3f}, {dex['ci95'][1]:.3f}] "
              f"({dex['n_sessions']} sessions, {dex['n_measurements']} measurements) {dex['note']}")
        rep["decay_exponent"] = dex
        lin = linearity(pr)
        print(f"5. linearity: V/I spread across amplitudes, worst {lin['max_rel_spread_V_over_I']}")
        rep["linearity"] = lin
        # residual figure
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
        d = np.array([r["distance_mm"] for r in pr])
        for name, e in errs.items():
            axs[0].semilogx(d, np.array(e["rel_errors"]) * 100, "o", ms=4, alpha=0.7, label=name)
        axs[0].axhline(0, color="0.5", lw=1)
        axs[0].set_xlabel("distance (mm)"); axs[0].set_ylabel("(measured - predicted) / predicted (%)")
        axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("residual vs distance, per model")
        V = np.array([r["plateau_V"] for r in pr]); P = np.array([r["V_pred"] for r in pr])
        B = np.array([r["band_V"] for r in pr])
        axs[1].errorbar(d, P * 1e3, yerr=B * 1e3, fmt="none", ecolor="#2a78d6", alpha=0.5,
                        label="committed prediction, 95% band")
        axs[1].loglog(d, V * 1e3, "k.", label="measured")
        axs[1].set_xlabel("distance (mm)"); axs[1].set_ylabel("potential (mV)")
        axs[1].legend(frameon=False, fontsize=8)
        axs[1].set_title(f"coverage {k}/{n} = {cov*100:.0f}%")
        fig.savefig(fig_dir / f"part{part}_comparison.png", dpi=200)
    report[f"part{part}"] = rep

sess = read_csv(sess_csv)
print("\nconductivity drift per session (start -> end)")
for s in sess:
    drift = (s["sigma_end"] - s["sigma_start"]) / s["sigma_start"]
    print(f"   session {s['session']}: {s['sigma_start']:.4f} -> {s['sigma_end']:.4f} S/m "
          f"({drift*100:+.2f}%; equivalent {-drift*100:+.2f}% in predicted potential)")
report["drift"] = [{"session": s["session"], "rel_drift": (s["sigma_end"] - s["sigma_start"]) / s["sigma_start"]}
                   for s in sess]
if recip_csv:
    rc = reciprocity(read_csv(recip_csv))
    print("reciprocity (swapped - forward) / forward")
    for r in rc:
        print(f"   session {r['session']}, {r['distance_mm']} mm: {r['rel_difference']*100:+.2f}%")
    report["reciprocity"] = rc
(out_dir / "comparison.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nwrote {out_dir.relative_to(ROOT)}/ and {fig_dir.relative_to(ROOT)}/")
