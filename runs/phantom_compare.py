"""Compare phantom measurements with the COMMITTED predictions (plan v16).

    python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 \\
        data/phantom/measurements.csv data/phantom/sessions.csv [data/phantom/reciprocity.csv]

Templates for the CSV files: notes/phantom_*_template.csv. Plateau voltages
come from runs/scope_plateau.py (or are read off the scope).

Reports, in this order (fixed in advance; the first is the headline):
  1. coverage: fraction of measurements inside the committed 95% band,
     separately for condition A, condition B and the Part 7 traverse
  2. the A - B difference: inside its band? the same at every position?
  3. the gradient ratio across the agar interface: measured vs committed
     finite-difference prediction (corrected to the measured conductivities),
     with the ideal ratio and the gap beside it
  4. condition B vs the idealized models: RMSE, relative error, paired
     Wilcoxon signed-rank (full model vs each), decay exponent with a
     bootstrap 95% CI (sessions, then positions)
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
from phantom_analysis import (read_csv, predict_rows, coverage, errors_by_model, wilcoxon_vs_full,
                              decay_exponent, reciprocity, linearity, difference_test,
                              gradient_ratio_test)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 3:
    sys.exit(__doc__)
pred_dir, meas_csv, sess_csv = Path(args[0]), args[1], args[2]
recip_csv = args[3] if len(args) > 3 else None
pred = json.loads((pred_dir / "predictions.json").read_text())
if pred.get("quick"):
    print("WARNING: these are QUICK predictions (coarse grids), not a committed set.")
if not pred.get("setup_file"):
    print("WARNING: these predictions used ASSUMED geometry, not a measured setup file.")

sessions = read_csv(sess_csv)
rows = predict_rows(read_csv(meas_csv), sessions, pred)
out_dir = ROOT / "results" / "phantom_comparison" / pred_dir.name
fig_dir = ROOT / "figures" / "phantom_comparison" / pred_dir.name
out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
report = {"predictions": str(pred_dir)}

def show_cov(name, rr):
    cov, k, n = coverage(rr)
    print(f"   {name:28s} {k}/{n} = {cov*100:.0f}% inside the committed 95% band")
    for r in rr:
        if not r["covered"]:
            print(f"      outside: session {r['session']}, position {r['position']} "
                  f"({r['distance_mm']:.1f} mm), {r.get('pulse_width_us')} us: measured "
                  f"{r['plateau_V']*1e3:.4f} mV, predicted {r['V_pred']*1e3:.4f} +- {r['band_V']*1e3:.4f} mV")
    return {"coverage": cov, "covered": k, "n": n}

print("1. COVERAGE")
groups = {"Part 6, condition A": [r for r in rows if int(r["part"]) == 6 and r["condition"] == "A"],
          "Part 6, condition B": [r for r in rows if int(r["part"]) == 6 and r["condition"] == "B"],
          "Part 7, traverse": [r for r in rows if int(r["part"]) == 7]}
report["coverage"] = {k: show_cov(k, v) for k, v in groups.items() if v}

dt, flat = difference_test(rows, sessions, pred)
if dt:
    c = np.mean([d["covered"] for d in dt])
    print(f"\n2. DIFFERENCE A - B: {sum(d['covered'] for d in dt)}/{len(dt)} = {c*100:.0f}% inside the band; "
          f"predicted {dt[0]['V_pred']/dt[0]['current_A']:.4g} V/A at session sigma")
    for sname, f in flat.items():
        if f["rel_spread"] is not None:
            print(f"   session {sname}: spread of (V_A - V_B)/I across positions {f['rel_spread']*100:.1f}% "
                  f"(predicted 0: the difference does not depend on position)")
    report["difference"] = {"rows": dt, "flatness": flat}

gt = gradient_ratio_test(rows, sessions, pred)
if gt:
    print("\n3. GRADIENT RATIO ACROSS THE AGAR INTERFACE")
    for g in gt:
        print(f"   session {g['session']}, stim {g['stim_index']}, {g['pulse_width_us']} us: measured "
              f"{g['R_meas']:.4f}, predicted {g['R_pred']:.4f} +- {g['U_rel']*g['R_pred']:.4f} "
              f"({'inside' if g['covered'] else 'OUTSIDE'}); ideal (measured sigmas) "
              f"{g['R_ideal_measured_sigmas']:.4f}, gap {g['gap_fd_vs_ideal']*100:+.1f}%; noise part of band "
              f"{g['U_noise_rel']*100:.1f}%")
    report["gradient_ratio"] = gt

B = groups["Part 6, condition B"]
if B:
    print("\n4. CONDITION B vs the idealized models")
    errs = errors_by_model(B)
    for name, e in errs.items():
        print(f"   {name:22s} RMSE {e['rmse_V']*1e3:8.4f} mV   median |rel. error| {e['median_abs_rel_error']*100:6.1f}%")
    report["errors_B"] = {k: {kk: v for kk, v in e.items() if kk != "rel_errors"} for k, e in errs.items()}
    w = wilcoxon_vs_full(B)
    for name, v in w.items():
        print(f"   Wilcoxon, full model vs {name:18s} " + (v["note"] if v["p"] is None else
              f"n = {v['n']}, full better in {v['full_model_better_in']}, p = {v['p']:.2g}"))
    report["wilcoxon_B"] = w
    dex = decay_exponent(B)
    print(f"   decay exponent (V_B ~ r^-n): n = {dex['exponent']:.3f}, 95% CI [{dex['ci95'][0]:.3f}, "
          f"{dex['ci95'][1]:.3f}] ({dex['n_sessions']} sessions) {dex['note']}")
    report["decay_exponent_B"] = dex
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    d = np.array([r["distance_mm"] for r in B])
    for name, e in errs.items():
        axs[0].semilogx(d, np.array(e["rel_errors"]) * 100, "o", ms=4, alpha=0.7, label=name)
    axs[0].axhline(0, color="0.5", lw=1)
    axs[0].set_xlabel("distance (mm)"); axs[0].set_ylabel("(measured - predicted) / predicted (%)")
    axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("condition B residuals, per model")
    for cond, c in (("A", "#2a78d6"), ("B", "#7b4fd0")):
        rr = groups[f"Part 6, condition {cond}"]
        if not rr:
            continue
        dd = np.array([r["distance_mm"] for r in rr])
        axs[1].errorbar(dd, [r["V_pred"] * 1e3 for r in rr], yerr=[r["band_V"] * 1e3 for r in rr],
                        fmt="none", ecolor=c, alpha=0.5)
        axs[1].loglog(dd, [r["plateau_V"] * 1e3 for r in rr], ".", color=c, label=f"condition {cond}")
    axs[1].set_xlabel("distance (mm)"); axs[1].set_ylabel("potential (mV)"); axs[1].legend(frameon=False)
    axs[1].set_title("measured (dots) vs committed prediction with 95% band")
    fig.savefig(fig_dir / "part6_comparison.png", dpi=200)

P6 = [r for r in rows if int(r["part"]) == 6]
if P6:
    lin = linearity(P6)
    print(f"\n5. linearity: worst spread of V/I across amplitudes {lin['max_rel_spread_V_over_I']}")
    report["linearity"] = lin
print("   conductivity drift per session (start -> end)")
for s in sessions:
    drift = (s["sigma_end"] - s["sigma_start"]) / s["sigma_start"]
    print(f"   session {s['session']}: {s['sigma_start']:.4f} -> {s['sigma_end']:.4f} S/m ({drift*100:+.2f}%)")
if recip_csv:
    rc = reciprocity(read_csv(recip_csv))
    for r in rc:
        print(f"   reciprocity, session {r['session']}, {r['distance_mm']} mm: {r['rel_difference']*100:+.2f}%")
    report["reciprocity"] = rc
(out_dir / "comparison.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nwrote {out_dir.relative_to(ROOT)}/ and {fig_dir.relative_to(ROOT)}/")
