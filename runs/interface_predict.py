"""Interface: fit a constant-phase element to the impedance sweep, then
predict the PULSE TRANSIENT blind from that fit. Commit and tag the output
BEFORE any pulse trace is opened (its own tag: phantom-interface-prediction-v1).

    python runs/interface_predict.py data/phantom/impedance.csv notes/phantom_setup.json \\
        results/phantom_predictions/phantom-prediction-v1 --sigma 1.52 \\
        --label phantom-interface-prediction-v1 [--layout E1_distance] [--position 5]
    python runs/interface_predict.py --demo      made-up sweep, testing only

Order of work on the day (so the prediction stays blind):
  1. impedance sweep, electrode in the cell and lifted out (plan step 56/57),
     with EXACTLY the wiring and probes that will be attached during pulses
  2. this script; commit; tag
  3. only then record the pulse traces; runs/interface_compare.py

Impedance CSV columns: condition (cell | lifted), repeat, frequency_Hz,
V_resistor, V_electrode, phase_deg (electrode voltage relative to the current).

What is predicted, each as a band (Monte Carlo over the fit's covariance, the
stray capacitance, the series resistor and the committed field prediction):
  the probe trace V(t) = (G / sigma) I(t) for every pulse width in the setup
  droop       fall of the plateau across the phase (the CPE charging)
  rise time   10-90% of the first edge (expected far below what the scope
              resolves: the prediction is an upper bound)
  node step and node growth   the voltage on the electrode side of the series
              resistor: the jump I R_s and its t^alpha growth -- measurable
              if a second scope channel is clipped there
  R_s         the fitted solution resistance against the field solver's
              committed disk-to-return resistance (no fitting between them)
The ideal-capacitor fit and ITS droop are reported beside the CPE's: that is
the comparison "CPE, not an ideal capacitor".
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phantom import load_setup
from phantom_analysis import read_csv
from scope import impedance_from_sweep
from transient import (z_cell, fit_cell, fit_stray, sample_fits, pulse_response, biphasic_segments,
                       transient_metrics)

def opt(name, default=None, cast=str):
    if name in sys.argv:
        k = sys.argv.index(name)
        v = cast(sys.argv[k + 1]); del sys.argv[k:k + 2]
        return v
    return default
DEMO = "--demo" in sys.argv
if DEMO:
    sys.argv.remove("--demo")
label = opt("--label", "interface_quick" if DEMO else "phantom-interface-prediction-v1")
sigma = opt("--sigma", None, float)
layout = opt("--layout", None)
position = opt("--position", 5, int)
args = sys.argv[1:]
if not DEMO and len(args) < 3:
    sys.exit(__doc__)
OUT = ROOT / "results" / "interface_predictions" / label
FIG = ROOT / "figures" / "interface_predictions" / label
if OUT.exists() and any(OUT.iterdir()) and not DEMO:
    sys.exit(f"{OUT.relative_to(ROOT)} already exists -- predictions are never overwritten. Use --label.")
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

# ---------------------------------------------------------------- inputs
if DEMO:
    setup = load_setup(None)
    f = np.array([10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
    Zw = 1 / (1j * 2 * np.pi * f * 80e-12)
    Z = 1 / (1 / z_cell(f, 650.0, 5e6, 1.5e-7, 0.82) + 1 / Zw)
    Z = Z * (1 + rng.normal(0, 0.01, len(f))) * np.exp(1j * np.radians(rng.normal(0, 0.5, len(f))))
    pq = ROOT / "results" / "phantom_predictions" / "phantom_quick" / "predictions.json"
    pred = json.loads(pq.read_text()) if pq.exists() else None
    sigma = sigma or setup["part6"]["sigma_nominal_S_per_m"]
    print("DEMO: made-up impedance sweep (R_s 650 ohm, Q 1.5e-7, alpha 0.82, stray 80 pF). Not a prediction.")
else:
    rows = read_csv(args[0]); setup = load_setup(args[1])
    pred = json.loads((Path(args[2]) / "predictions.json").read_text())
    if sigma is None:
        sys.exit("--sigma is required: the saline conductivity measured in this session (S/m)")
    Rser = setup["instrument"]["series_resistor_ohm"]
    def Z_of(cond):
        rs = [r for r in rows if r["condition"] == cond]
        fs = sorted(set(r["frequency_Hz"] for r in rs))
        return np.array(fs), np.array([np.mean([impedance_from_sweep(
            r["frequency_Hz"], r["V_resistor"], r["V_electrode"], r["phase_deg"], Rser)
            for r in rs if r["frequency_Hz"] == fq]) for fq in fs])
    f, Z = Z_of("cell")
    Zw = None
    if any(r["condition"] == "lifted" for r in rows):
        fw, Zw_ = Z_of("lifted")
        Zw = np.interp(np.log(f), np.log(fw), Zw_.real) + 1j * np.interp(np.log(f), np.log(fw), Zw_.imag)
    else:
        print("WARNING: no 'lifted' rows -- the stray capacitance is taken as 0 and its effect is not in the band.")

inst, pulse, unc = setup["instrument"], setup["pulse"], setup["uncertainties"]
Rser = inst["series_resistor_ohm"]
fit = fit_cell(f, Z, Zw)
cap = fit_cell(f, Z, Zw, ideal_capacitor=True)
stray = fit_stray(f, Zw) if Zw is not None else {"C_stray_F": 0.0, "C_stray_rel_sd": 0.0, "R_leak_ohm": np.inf}
print(f"\nCPE fit:        R_s {fit['R_s_ohm']:.1f} ohm, R_ct {fit['R_ct_ohm']:.3g} ohm, Q {fit['Q']:.3g}, "
      f"alpha {fit['alpha']:.3f}; residual {fit['rms_log_residual']:.4f} (log|Z|), {fit['rms_phase_residual_deg']:.2f} deg")
print(f"ideal capacitor: R_s {cap['R_s_ohm']:.1f} ohm, C {cap['Q']*1e9:.2f} nF; residual "
      f"{cap['rms_log_residual']:.4f} (log|Z|), {cap['rms_phase_residual_deg']:.2f} deg "
      f"-> the CPE fits {cap['rms_log_residual']/max(fit['rms_log_residual'],1e-12):.1f}x closer")
print(f"stray (electrode lifted): {stray['C_stray_F']*1e12:.1f} pF")

# field prediction for the probe position
G = G_lo = G_hi = None; Rsol_pred = None
if pred is not None:
    lay = layout or next(iter(pred["experiments"]))
    ex = pred["experiments"][lay]
    row = ex["rows"][position]
    G, G_lo, G_hi = row["G_A_per_m"], row["G_A_lo_per_m"], row["G_A_hi_per_m"]
    Rsol_pred = ex["R_disk_to_return_x_sigma_by_grid"][-1] / sigma
    print(f"probe: layout {lay}, position {position} ({row['distance_mm']} mm), condition A: "
          f"G = {G:.2f} 1/m [{G_lo:.2f}, {G_hi:.2f}] -> {G/sigma/1e3:.4f} mV per uA at {sigma} S/m")
    print(f"R_s: fitted {fit['R_s_ohm']:.1f} ohm; field solver (committed) {Rsol_pred:.1f} ohm at {sigma} S/m "
          f"-> ratio {fit['R_s_ohm']/Rsol_pred:.3f}")
else:
    G, G_lo, G_hi = 31.0, 30.0, 32.0
    print("no field prediction found: using a placeholder G = 31 1/m for the demo")

# ---------------------------------------------------------------- transient
A = pulse["generator_amplitude_V"]; T = 1.0 / pulse["repetition_Hz"]
t0 = 0.1 * T
N_MC = 60 if DEMO else 300
cells = sample_fits(fit, N_MC, rng)
out = {"label": label, "demo": DEMO, "created": time.strftime("%Y-%m-%d %H:%M"), "sigma_S_per_m": sigma,
       "fit_cpe": fit, "fit_ideal_capacitor": cap, "stray": stray, "series_resistor_ohm": Rser,
       "R_s_field_solver_ohm": Rsol_pred, "impedance_frequencies_Hz": f.tolist(),
       "probe": {"G_per_m": G, "G_lo_per_m": G_lo, "G_hi_per_m": G_hi, "position": position},
       "pulse": pulse, "widths": {}}
fig, axs = plt.subplots(1, len(pulse["phase_widths_us"]), figsize=(5 * len(pulse["phase_widths_us"]), 4.2),
                        layout="constrained", squeeze=False)
for ax, w_us in zip(axs[0], pulse["phase_widths_us"]):
    ph = w_us * 1e-6
    dt = min(ph / 2000, 0.25e-6)
    seg = biphasic_segments(A, ph, t0, pulse["interphase_us"] * 1e-6)
    def run(cell, cs, R):
        return pulse_response(seg, T, dt, R, cell, cs, pulse.get("generator_rise_time_s", 0.0))
    best = (fit["R_s_ohm"], fit["R_ct_ohm"], fit["Q"], fit["alpha"])
    t, I0, Vn0 = run(best, stray["C_stray_F"], Rser)
    _, Ic, _ = run((cap["R_s_ohm"], cap["R_ct_ohm"], cap["Q"], 1.0), stray["C_stray_F"], Rser)
    sel = (t >= t0 - 0.25 * ph) & (t <= t0 + 2.5 * ph)
    m0 = transient_metrics(t, I0, ph, t0); mc = transient_metrics(t, Ic, ph, t0)
    mets, waves = [], []
    for cell in cells:
        cs = stray["C_stray_F"] * (1 + rng.normal(0, max(stray["C_stray_rel_sd"], 0.05)))
        R = Rser * (1 + rng.normal(0, unc["current_rel_sd"]))
        _, I, Vn = run(cell, max(cs, 0.0), R)
        g = rng.normal(G, (G_hi - G_lo) / 3.92)
        m = transient_metrics(t, I, ph, t0)
        mn = transient_metrics(t, Vn, ph, t0)
        step = float(np.interp(t0 + 0.02 * ph, t, Vn) - np.interp(t0 - 0.02 * ph, t, Vn))
        grow = float(np.interp(t0 + 0.9 * ph, t, Vn) - np.interp(t0 + 0.1 * ph, t, Vn))
        mets.append([m["level"] * g / sigma, m["droop"], m["rise_s"], step, grow, m["level"]])
        waves.append(I[sel] * g / sigma)
    mets = np.array(mets); waves = np.array(waves)
    pc = lambda k: [float(v) for v in np.percentile(mets[:, k], [2.5, 50, 97.5])]
    rec = {"plateau_V": pc(0), "droop": pc(1), "rise_s": pc(2), "node_step_V": pc(3), "node_growth_V": pc(4),
           "plateau_current_A": pc(5), "droop_ideal_capacitor": mc["droop"], "droop_best_fit": m0["droop"]}
    out["widths"][f"{w_us:g}"] = rec
    lo, med, hi = np.percentile(waves, [2.5, 50, 97.5], axis=0)
    step_ = max(1, int(sel.sum() // 4000))
    np.savetxt(OUT / f"probe_trace_{w_us:g}us.csv",
               np.column_stack([(t[sel] - t0)[::step_], med[::step_], lo[::step_], hi[::step_],
                                (Vn0[sel])[::step_]]),
               delimiter=",", header="time_from_edge_s,V_probe_V,V_probe_lo,V_probe_hi,V_node_V", comments="")
    print(f"\n{w_us:g} us per phase, {A:g} V into {Rser/1e3:g} kOhm:")
    print(f"  probe plateau {rec['plateau_V'][1]*1e3:.4f} mV [{rec['plateau_V'][0]*1e3:.4f}, {rec['plateau_V'][2]*1e3:.4f}] "
          f"at {rec['plateau_current_A'][1]*1e6:.2f} uA")
    print(f"  DROOP across the phase {rec['droop'][1]*100:.2f}% [{rec['droop'][0]*100:.2f}, {rec['droop'][2]*100:.2f}]"
          f"   (ideal-capacitor fit would give {mc['droop']*100:.2f}%)")
    print(f"  rise time {rec['rise_s'][1]*1e9:.0f} ns [{rec['rise_s'][0]*1e9:.0f}, {rec['rise_s'][2]*1e9:.0f}] "
          f"-> faster than a school scope resolves; prediction: no slower than the scope's own edge")
    print(f"  electrode node: step {rec['node_step_V'][1]*1e3:.1f} mV, then growth {rec['node_growth_V'][1]*1e3:.1f} mV "
          f"[{rec['node_growth_V'][0]*1e3:.1f}, {rec['node_growth_V'][2]*1e3:.1f}] between 10% and 90% of the phase")
    x = (t[sel] - t0) * 1e6
    ax.fill_between(x, lo * 1e3, hi * 1e3, color="#2a78d6", alpha=0.3, lw=0)
    ax.plot(x, med * 1e3, color="#2a78d6", lw=1.5, label="CPE prediction (95% band)")
    ax.plot(x, Ic[sel] * G / sigma * 1e3, color="#eb6834", lw=1, ls="--", label="ideal-capacitor fit")
    ax.set_xlabel("time from the pulse edge (us)"); ax.set_ylabel("probe potential (mV)")
    ax.set_title(f"{w_us:g} us per phase: droop {rec['droop'][1]*100:.1f}%", fontsize=10)
    ax.grid(True, color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"Blind pulse-transient prediction from the impedance sweep [{label}]" + (" -- DEMO" if DEMO else ""))
fig.savefig(FIG / "interface_prediction.png", dpi=200)
(OUT / "prediction.json").write_text(json.dumps(out, indent=2))
print(f"\nwrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
if not DEMO:
    print(f"NEXT, BEFORE OPENING ANY PULSE TRACE: git add {args[0]} {OUT.relative_to(ROOT)} {FIG.relative_to(ROOT)} "
          f"&& git commit -m 'Interface: blind pulse-transient prediction' && git tag {label} && git push "
          f"&& git push --tags")
