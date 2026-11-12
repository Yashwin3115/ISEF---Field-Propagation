"""Plan v16: stimulator output impedance sweep. The 100-220 mV peak artifact
was an UPPER bound (ideal voltage source). Here the source has an output
resistance R_src from 0 to 100 kOhm (ASSUMED range -- no published value for
the MaxOne stimulation buffer was found; 10 kOhm is the value used in the
Aim 2 synthetic world). Reported against R_src / |Z_electrode| (|Z| at 1 kHz,
450 kOhm, itself the middle of the published 300-600 kOhm range), so the
result does not depend on that one number.

For each R_src: peak current, interface time constant, charge per phase, and
the peak artifact at 1, 10 and 100 pitches -- both instantaneous and as the
amplifier samples it (50 us average). The two ends of the range define the
reported BAND for the peak artifact; runs/saturation_map.py --rsrc <ohm>
recomputes the saturation map at each end.

    python runs/stimulator_sweep.py          (~2 min, ~4 GB)
    python runs/stimulator_sweep.py --quick
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
from maxone import transfer_run, PITCH, PROVISIONAL, um, mm
from interface import capacitance_from_impedance, pulse_summary, biphasic, circuit_current

QUICK = "--quick" in sys.argv
NAME = "stimulator_sweep_quick" if QUICK else "stimulator_sweep"
OUT = ROOT / "results" / "runs" / NAME; FIG = ROOT / "figures" / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
GRID = dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK else \
       dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm)
AMP, PHASE = 0.5, 200e-6
Z_EL = 450e3
R_SRC = np.array([0, 1e3, 3e3, 10e3, 30e3, 100e3])
DIST_P = [1, 10, 100]

r = transfer_run({}, verbose=False, **GRID)
R_sol = r["R_solution_ohm"]
C = capacitance_from_impedance(Z_EL)
c0, r0 = PROVISIONAL["stim_col_row"]
row = (r["row"] == r0) & (r["col"] > c0)
Zd = {n: float(r["transfer_ohm"][row][np.argmin(np.abs(r["distance_m"][row] / PITCH - n))]) for n in DIST_P}
dt = 0.05e-6
t = np.arange(0, 2 * PHASE + 0.2e-3, dt)
rows = []
print(f"solution resistance {R_sol/1e3:.1f} kOhm; |Z_electrode| {Z_EL/1e3:.0f} kOhm -> C {C*1e9:.3f} nF")
print(f"{'R_src':>8} {'R_src/|Z|':>9} {'tau us':>7} {'I peak uA':>9} {'Q pC':>7} "
      + " ".join(f"{'pk@'+str(n)+'p mV':>11} {'(50us)':>7}" for n in DIST_P))
for Rs in R_SRC:
    s = pulse_summary(AMP, PHASE, R_sol, C, R_buf=Rs)
    i = circuit_current(t, biphasic(AMP, PHASE), R_sol, C, Rs)
    k = np.floor(t / 50e-6).astype(int)
    i50 = np.bincount(k, weights=i) / np.bincount(k)
    rec = {"R_src_ohm": float(Rs), "R_src_over_Z_electrode": float(Rs / Z_EL),
           "tau_us": s["tau_s"] * 1e6, "peak_current_uA": s["peak_current_A"] * 1e6,
           "charge_phase1_pC": s["charge_phase1_C"] * 1e12}
    for n in DIST_P:
        rec[f"peak_artifact_{n}p_mV"] = Zd[n] * s["peak_current_A"] * 1e3
        rec[f"peak_artifact_{n}p_sampled_mV"] = Zd[n] * float(np.max(np.abs(i50))) * 1e3
    rows.append(rec)
    print(f"{Rs/1e3:7.0f}k {Rs/Z_EL:9.4f} {rec['tau_us']:7.1f} {rec['peak_current_uA']:9.2f} "
          f"{rec['charge_phase1_pC']:7.1f} " + " ".join(
              f"{rec[f'peak_artifact_{n}p_mV']:11.3f} {rec[f'peak_artifact_{n}p_sampled_mV']:7.3f}" for n in DIST_P))
band = {f"{n} pitches": [rows[-1][f"peak_artifact_{n}p_mV"], rows[0][f"peak_artifact_{n}p_mV"]] for n in DIST_P}
print("\nPEAK ARTIFACT BAND (R_src 100 kOhm .. 0): " + "; ".join(
    f"{k}: {v[0]:.3g}-{v[1]:.3g} mV" for k, v in band.items()))
print(f"charge per phase is C V = {C*AMP*1e12:.0f} pC at every R_src (the capacitor charges fully: "
      f"tau << 200 us); the source resistance spreads the same charge over a longer, lower spike")
json.dump({"grid": GRID, "R_solution_ohm": R_sol, "Z_electrode_ohm": Z_EL, "C_nF": C * 1e9,
           "transfer_ohm": Zd, "rows": rows, "peak_artifact_band_mV": band,
           "assumption": "R_src 0-100 kOhm assumed; no published value found"},
          open(OUT / "summary.json", "w"), indent=2)
fig, ax = plt.subplots(figsize=(7, 4.3), layout="constrained")
x = np.maximum(R_SRC / Z_EL, 1e-4)
for n, c in zip(DIST_P, ["#c2410c", "#2a78d6", "#1baf7a"]):
    ax.loglog(x, [q[f"peak_artifact_{n}p_mV"] for q in rows], "o-", color=c, lw=2, label=f"{n} pitch(es), instantaneous")
    ax.loglog(x, [q[f"peak_artifact_{n}p_sampled_mV"] for q in rows], "o--", color=c, lw=1.2, label=f"{n}, 50 us sampled")
ax.set_xlabel("R_source / |Z_electrode|  (leftmost point: ideal source, plotted at 1e-4)")
ax.set_ylabel("peak artifact (mV)")
ax.set_title("Peak artifact vs stimulator output resistance (500 mV, |Z| 450 kOhm)", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=7.5)
fig.savefig(FIG / "peak_artifact_vs_source.png", dpi=180)
print(f"wrote {OUT.relative_to(ROOT)}/")
