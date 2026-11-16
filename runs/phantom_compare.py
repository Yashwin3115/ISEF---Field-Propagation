"""Compare phantom measurements with the COMMITTED predictions and apply the
COMMITTED pass criteria (plan v16 + the Nov 15 experiment list).

    python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 \\
        data/phantom/measurements.csv data/phantom/sessions.csv [data/phantom/reciprocity.csv]

Templates for the CSV files: notes/phantom_*_template.csv. Plateau voltages
and noise floors come from runs/scope_plateau.py.

Reports, in this order (fixed in advance; the verdicts at the end use only
the rules in pass_criteria.json, written before any measurement):
  1. coverage of every reading by its committed band: each layout, conditions A and B, traverse
  2. full model vs the point-source formulas (condition B), at the positions
     that were predicted to tell them apart
  3. the A - B difference: inside its band? the same at every position?
  4. the plate ratio and the return ratios
  5. two saline concentrations: do the readings collapse?
  6. the gradient ratio across the agar interface
  7. linearity in current, reciprocity, conductivity drift
  8. VERDICTS
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
                              gradient_ratio_test, ratio_test, sigma_collapse_test)
from phantom_criteria import k_min, closer_test, verdict, RULES

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 3:
    sys.exit(__doc__)
pred_dir, meas_csv, sess_csv = Path(args[0]), args[1], args[2]
recip_csv = args[3] if len(args) > 3 else None
pred = json.loads((pred_dir / "predictions.json").read_text())
crit = json.loads((pred_dir / "pass_criteria.json").read_text())
if pred.get("quick"):
    print("WARNING: these are QUICK predictions (coarse grids), not a committed set.")
if not pred.get("setup_file"):
    print("WARNING: these predictions used ASSUMED geometry, not a measured setup file.")

sessions = read_csv(sess_csv)
rows = predict_rows(read_csv(meas_csv), sessions, pred)
out_dir = ROOT / "results" / "phantom_comparison" / pred_dir.name
fig_dir = ROOT / "figures" / "phantom_comparison" / pred_dir.name
out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
report = {"predictions": str(pred_dir), "rules": crit["rules"]}
verdicts = []

def show_cov(name, rr):
    cov, k, n = coverage(rr)
    need = k_min(n)
    ok = k >= need
    print(f"   {name:34s} {k}/{n} = {cov*100:.0f}% inside the committed band "
          f"(needed >= {need}): {'PASS' if ok else 'NOT PASSED'}")
    for r in rr:
        if not r["covered"]:
            print(f"      outside: session {r['session']}, position {r['position']} "
                  f"({r['distance_mm']:.1f} mm), {r.get('pulse_width_us')} us: measured "
                  f"{r['plateau_V']*1e3:.4f} mV, band [{r['band_lo_V']*1e3:.4f}, {r['band_hi_V']*1e3:.4f}] mV")
    return {"coverage": cov, "covered": k, "n": n, "needed": need, "pass": bool(ok)}

print("1. COVERAGE")
P6 = [r for r in rows if int(r["part"]) == 6]
layouts = list(dict.fromkeys(r["layout"] for r in P6))
groups = {}
for lay in layouts:
    for c in ("A", "B"):
        groups[f"{lay}, condition {c}"] = [r for r in P6 if r["layout"] == lay and r["condition"] == c]
groups["Part 7, traverse"] = [r for r in rows if int(r["part"]) == 7]
report["coverage"] = {k: show_cov(k, v) for k, v in groups.items() if v}

print("\n2. FULL MODEL vs POINT-SOURCE FORMULAS (condition B)")
report["models_B"] = {}
for lay in layouts:
    B = groups[f"{lay}, condition B"]
    if not B:
        continue
    errs = errors_by_model(B)
    print(f"   {lay}:")
    for name, e in errs.items():
        print(f"      {name:22s} RMSE {e['rmse_V']*1e3:8.4f} mV   median |rel. error| "
              f"{e['median_abs_rel_error']*100:6.1f}%")
    rep = {"errors": {k: {kk: v for kk, v in e.items() if kk != "rel_errors"} for k, e in errs.items()}}
    full_pass = report["coverage"][f"{lay}, condition B"]["pass"]
    for name, c in crit["experiments"].get(lay, {}).get("simple_models_condition_B", {}).items():
        sel = [r for r in B if int(r["position"]) in c["discriminating_positions"]]
        meas = np.array([r["plateau_V"] for r in sel])
        ct = closer_test(np.abs(meas - np.array([r["V_models"]["full model"] for r in sel])),
                         np.abs(meas - np.array([r["V_models"][name] for r in sel])))
        fails = ct["simple_model_fails"] if c["testable"] else None
        rep[name] = ct
        if c["testable"] and ct["n"]:
            print(f"      vs {name}: at its {len(c['discriminating_positions'])} discriminating positions, "
                  f"full model closer in {ct['full_closer_fraction']*100:.0f}% of {ct['n']} readings"
                  + (f", p = {ct['p']:.2g}" if ct["p"] is not None else "") +
                  f" -> {'FAILS' if fails else 'does not fail'}")
        else:
            print(f"      vs {name}: not testable (fewer than {RULES['min_discriminating']} discriminating "
                  f"positions, stated before measuring)" if not c["testable"] else f"      vs {name}: no readings")
        verdicts.append((f"{lay}: full model vs {name}", verdict(full_pass, fails)))
    if lay == layouts[0]:
        dex = decay_exponent([r for r in B if r.get("well_above_zero")])
        print(f"      decay exponent (V_B ~ r^-n): n = {dex['exponent']:.3f}, 95% CI [{dex['ci95'][0]:.3f}, "
              f"{dex['ci95'][1]:.3f}] ({dex['n_sessions']} sessions) {dex['note']}")
        rep["decay_exponent"] = dex
    report["models_B"][lay] = rep

dt, flat = difference_test(rows, sessions, pred)
if dt:
    print("\n3. DIFFERENCE A - B")
    need = k_min(len(dt)); k = sum(d["covered"] for d in dt)
    flat_ok = all(f["rel_spread"] is None or f["rel_spread"] <= f["allowed_rel_spread"] for f in flat.values())
    print(f"   {k}/{len(dt)} inside the band (needed >= {need}): {'PASS' if k >= need else 'NOT PASSED'}")
    for sname, f in flat.items():
        if f["rel_spread"] is not None:
            print(f"   session / layout {sname}: spread of (V_A - V_B)/I across positions "
                  f"{f['rel_spread']*100:.1f}% (predicted 0; allowed {f['allowed_rel_spread']*100:.1f}%)")
    report["difference"] = {"rows": dt, "flatness": flat, "pass": bool(k >= need and flat_ok)}
    verdicts.append(("A - B difference (value and flatness)",
                     "PASS" if k >= need and flat_ok else "NOT PASSED"))

print("\n4. RATIOS BETWEEN LAYOUTS (condition B)")
report["ratios"] = {}
def do_ratio(title, table, cblock, num, den, simple):
    rt = ratio_test(rows, pred, table, num, den, simple)
    if not rt:
        return
    k, n = sum(q["covered"] for q in rt), len(rt)
    need = k_min(n)
    full_pass = k >= need
    sel = [q for q in rt if q["position"] in cblock["discriminating_positions"]]
    ct = closer_test([q["e_full"] for q in sel], [q["e_simple"] for q in sel])
    fails = ct["simple_model_fails"] if cblock["testable"] else None
    print(f"   {title}: {k}/{n} covered (needed >= {need}): {'PASS' if full_pass else 'NOT PASSED'}; "
          f"against 'always {simple:g}': " + (f"full model closer in {ct['full_closer_fraction']*100:.0f}% of "
          f"{ct['n']} -> {'FAILS' if fails else 'does not fail'}" if cblock["testable"] and ct["n"] else "not testable"))
    for q in rt:
        print(f"      {q['distance_mm']:5.1f} mm: measured {q['R_meas']:.3f}, predicted {q['R_pred']:.3f} "
              f"[{q['band'][0]:.3f}, {q['band'][1]:.3f}] {'' if q['covered'] else 'OUTSIDE'}")
    report["ratios"][title] = {"rows": rt, "pass": bool(full_pass), "closer": ct}
    verdicts.append((title, verdict(full_pass, fails)))
D = pred.get("derived", {})
if "plate_ratio" in D and "plate_ratio" in crit.get("derived", {}):
    do_ratio("plate ratio (plate / no plate)", D["plate_ratio"]["condition_B"], crit["derived"]["plate_ratio"],
             "E1_distance", "E2_no_plate", 2.0)
for n_, cb in crit.get("derived", {}).get("return_ratio", {}).items():
    do_ratio(f"return ratio ({n_} / far wall)", D["return_ratio"][n_], cb, n_, "E1_distance", 1.0)

sc = sigma_collapse_test(rows, pred)
if sc:
    k, n = sum(q["covered"] for q in sc), len(sc)
    need = k_min(n)
    c = np.array([q["collapse_ratio"] for q in sc])
    print(f"\n5. TWO SALINE CONCENTRATIONS: (V sigma / I) ratio between sessions, {n} pairs: median "
          f"{np.median(c):.3f}, range {c.min():.3f}-{c.max():.3f} (predicted 1); {k}/{n} inside the band "
          f"(needed >= {need}): {'PASS' if k >= need else 'NOT PASSED'}")
    report["sigma_collapse"] = {"rows": sc, "pass": bool(k >= need)}
    verdicts.append(("conductivity collapse", "PASS" if k >= need else "NOT PASSED"))

gt = gradient_ratio_test(rows, sessions, pred)
if gt:
    print("\n6. GRADIENT RATIO ACROSS THE AGAR INTERFACE")
    for g in gt:
        print(f"   session {g['session']}, stim {g['stim_index']}, {g['pulse_width_us']} us: measured "
              f"{g['R_meas']:.4f}, predicted {g['R_pred']:.4f} [{g['band'][0]:.4f}, {g['band'][1]:.4f}] "
              f"({'inside' if g['covered'] else 'OUTSIDE'}); ideal (measured sigmas) "
              f"{g['R_ideal_measured_sigmas']:.4f}, gap {g['gap_fd_vs_ideal']*100:+.1f}%; 'no interface' "
              f"{g['uniform_medium_ratio']} {'excluded' if g['uniform_medium_excluded'] else 'NOT excluded'}")
    report["gradient_ratio"] = gt
    k, n = sum(g["covered"] for g in gt), len(gt)
    exc = [g["uniform_medium_excluded"] for g in gt if g["uniform_medium_excluded"] is not None]
    verdicts.append(("Part 7 gradient ratio", verdict(k >= k_min(n), (all(exc) if exc else None))))

print("\n7. CHECKS")
if P6:
    lin = linearity(P6)
    print(f"   linearity: worst spread of V/I across amplitudes {lin['max_rel_spread_V_over_I']}")
    report["linearity"] = lin
for s in sessions:
    drift = (s["sigma_end"] - s["sigma_start"]) / s["sigma_start"]
    print(f"   session {s['session']}: conductivity {s['sigma_start']:.4f} -> {s['sigma_end']:.4f} S/m ({drift*100:+.2f}%)")
if recip_csv:
    rc = reciprocity(read_csv(recip_csv), RULES["reciprocity_floor_rel"])
    for r in rc:
        print(f"   reciprocity, session {r['session']} {r['layout']}, {r['distance_mm']} mm: "
              f"{r['rel_difference']*100:+.2f}% (allowed {r['allowed']*100:.1f}%) {'PASS' if r['pass'] else 'NOT PASSED'}")
    report["reciprocity"] = rc
    verdicts.append(("reciprocity", "PASS" if all(r["pass"] for r in rc) else "NOT PASSED"))

print("\n8. VERDICTS (rules fixed before measurement: notes/phantom_pass_criteria.md)")
for lay in layouts:
    for c in ("A", "B"):
        g = report["coverage"].get(f"{lay}, condition {c}")
        if g:
            print(f"   {lay}, condition {c}: full model {'PASSES' if g['pass'] else 'DOES NOT PASS'} "
                  f"({g['covered']}/{g['n']} covered)")
for name, v in verdicts:
    print(f"   {name}: {v}")
report["verdicts"] = verdicts

# ---- figure: every layout, measured vs band
if P6:
    fig, axs = plt.subplots(1, len(layouts), figsize=(5.2 * len(layouts), 4.3), layout="constrained", squeeze=False)
    for ax, lay in zip(axs[0], layouts):
        for cond, c in (("A", "#2a78d6"), ("B", "#7b4fd0")):
            rr = [r for r in groups[f"{lay}, condition {cond}"] if r["V_pred"] > 0 and r["plateau_V"] > 0]
            if not rr:
                continue
            dd = np.array([r["distance_mm"] for r in rr]); I = np.array([r["current_A"] for r in rr])
            vp = np.array([r["V_pred"] for r in rr]) / I
            ax.errorbar(dd, vp * 1e-3, yerr=[(vp - np.array([r["band_lo_V"] for r in rr]) / I) * 1e-3,
                                             (np.array([r["band_hi_V"] for r in rr]) / I - vp) * 1e-3],
                        fmt="none", ecolor=c, alpha=0.5)
            ax.loglog(dd, np.array([r["plateau_V"] for r in rr]) / I * 1e-3, ".", color=c, label=f"condition {cond}")
        ax.set_xlabel("distance (mm)"); ax.set_ylabel("mV per uA"); ax.set_title(lay, fontsize=10)
        ax.legend(frameon=False, fontsize=8); ax.grid(True, which="both", color="0.92", lw=0.5)
    fig.suptitle("measured (dots) vs committed prediction band")
    fig.savefig(fig_dir / "part6_comparison.png", dpi=200)
(out_dir / "comparison.json").write_text(json.dumps(report, indent=2, default=str))
print(f"\nwrote {out_dir.relative_to(ROOT)}/ and {fig_dir.relative_to(ROOT)}/")
