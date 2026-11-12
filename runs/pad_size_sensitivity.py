"""Plan v16: electrode-size sensitivity. The MaxOne User Guide gives
8.75 x 12 um; Ronchi et al. 2019 give 5 x 9 um^2 for an array with the same
electrode count, pitch and sensing area (probably a chip variant; unresolved).

The array prediction is rerun at both sizes (top-plate return, provisional
settings) and the consequences are traced through:
  1. transfer resistance vs distance (ratio small pad / large pad)
  2. the interface: solution resistance, peak current, charge per phase and
     peak artifact. The 5 x 9 um impedance is not published, so two cases:
     (a) same |Z| (300-600 kOhm) -> same capacitance
     (b) capacitance scaled with pad area (45 / 105 um^2)
  3. saturation: fraction of the array ever saturated at +-1, 3, 10, 30 mV
  4. Aim 2: can the large-pad template fit the small-pad pattern with its
     scale + offset? (residual by distance band)
  5. Aim 3: thresholds of the uniform 1 um axon at offsets 0-20 um

    python runs/pad_size_sensitivity.py         (~10-15 min, ~4 GB)
    python runs/pad_size_sensitivity.py --quick
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import transfer_run, CHIPS, PITCH, PROVISIONAL, electrode_xy, um, mm
from interface import capacitance_from_impedance, pulse_summary, biphasic, circuit_current
from neuron import Axon, MainenMembrane, straight_axon_points, thresholds

QUICK = "--quick" in sys.argv
NAME = "pad_size_quick" if QUICK else "pad_size"
OUT = ROOT / "results" / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)
GRID = dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK else \
       dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm)
CH = {"8.75 x 12 um (User Guide)": "MaxOne", "5 x 9 um (Ronchi 2019)": "MaxOne (5x9, Ronchi 2019)"}
AMP, PHASE = 0.5, 200e-6
Z1K = 450e3
STEPS = [1, 2, 5, 10, 20, 50, 100]
t0 = time.time()
sx, sy = electrode_xy(*PROVISIONAL["stim_col_row"])
c0, r0 = PROVISIONAL["stim_col_row"]
res, phi_box = {}, {}
for name, chip in CH.items():
    r = transfer_run({"chip": chip}, verbose=False, **GRID)
    res[name] = r
    x, y, z = r["x_grid"], r["y_grid"], r["z_grid"]
    bx = np.flatnonzero((x > sx - 1.1 * mm) & (x < sx + 1.1 * mm)); by = np.flatnonzero((y > sy - 1.1 * mm) & (y < sy + 1.1 * mm))
    bz = np.flatnonzero(z < 40 * um)
    bx = np.arange(max(bx[0] - 2, 0), min(bx[-1] + 3, len(x))); by = np.arange(max(by[0] - 2, 0), min(by[-1] + 3, len(y)))
    bz = np.arange(0, min(bz[-1] + 3, len(z)))
    phi_box[name] = RegularGridInterpolator((x[bx], y[by], z[bz]), r["phi_per_A"][np.ix_(bx, by, bz)], method="cubic")
    del r["phi_per_A"]
    print(f"{name}: solved ({time.time()-t0:.0f} s); solution resistance {r['R_solution_ohm']/1e3:.1f} kOhm")

big, small = list(CH)
out = {"grid": GRID, "quick": QUICK}
row = (res[big]["row"] == r0) & (res[big]["col"] > c0)
d_p = res[big]["distance_m"][row] / PITCH
pick = [int(np.argmin(np.abs(d_p - n))) for n in STEPS]
Zb, Zs = res[big]["transfer_ohm"][row][pick], res[small]["transfer_ohm"][row][pick]
out["transfer"] = {f"{n} pitches": {"large_ohm": float(a), "small_ohm": float(b), "ratio": float(b / a)}
                   for n, a, b in zip(STEPS, Zb, Zs)}
print("\n1. transfer resistance (small / large pad), same current:")
print("   " + "  ".join(f"{n}p {b/a:.3f}" for n, a, b in zip(STEPS, Zb, Zs)))

print("\n2. interface (500 mV, 200 us/phase, |Z| 450 kOhm for the large pad):")
area = {big: 8.75 * 12.0, small: 5.0 * 9.0}
C_big = capacitance_from_impedance(Z1K)
cases = {"large pad": (big, C_big), "small pad, same |Z|": (small, C_big),
         "small pad, C scaled with area": (small, C_big * area[small] / area[big])}
out["interface"] = {}
for cname, (pad, C) in cases.items():
    s = pulse_summary(AMP, PHASE, res[pad]["R_solution_ohm"], C)
    Z1 = res[pad]["transfer_ohm"][row][pick[0]]; Z10 = res[pad]["transfer_ohm"][row][pick[3]]
    rec = {"R_solution_kOhm": res[pad]["R_solution_ohm"] / 1e3, "C_nF": C * 1e9, "tau_us": s["tau_s"] * 1e6,
           "peak_current_uA": s["peak_current_A"] * 1e6, "charge_pC": s["charge_phase1_C"] * 1e12,
           "peak_artifact_1p_mV": Z1 * s["peak_current_A"] * 1e3, "peak_artifact_10p_mV": Z10 * s["peak_current_A"] * 1e3}
    out["interface"][cname] = rec
    print(f"   {cname:31s} R_sol {rec['R_solution_kOhm']:5.1f} kOhm, tau {rec['tau_us']:5.1f} us, "
          f"I peak {rec['peak_current_uA']:5.1f} uA, Q {rec['charge_pC']:5.0f} pC, peak at 1 pitch "
          f"{rec['peak_artifact_1p_mV']:6.1f} mV, at 10 {rec['peak_artifact_10p_mV']:5.1f} mV")

print("\n3. fraction of the array ever saturated (instantaneous peak, ideal source):")
out["saturation"] = {}
for cname, (pad, C) in cases.items():
    Ipk = pulse_summary(AMP, PHASE, res[pad]["R_solution_ohm"], C)["peak_current_A"]
    Zall = np.abs(res[pad]["transfer_ohm"][res[pad]["distance_m"] > 0])
    fr = {f"+-{m} mV": float(np.mean(Zall * Ipk > m * 1e-3)) for m in (1, 3, 10, 30)}
    out["saturation"][cname] = fr
    print(f"   {cname:31s} " + "  ".join(f"{k} {v*100:5.1f}%" for k, v in fr.items()))

print("\n4. Aim 2: large-pad template (scale + offset) fitted to the small-pad pattern:")
Zl_all = res[big]["transfer_ohm"]; Zs_all = res[small]["transfer_ohm"]
use = res[big]["distance_m"] > 0
X = np.column_stack([Zl_all[use], np.ones(use.sum())])
coef, *_ = np.linalg.lstsq(X, Zs_all[use], rcond=None)
rel = (X @ coef - Zs_all[use]) / np.abs(Zs_all[use])
dd = res[big]["distance_m"][use] / PITCH
out["aim2_template_fit"] = {"scale": float(coef[0]), "offset_ohm": float(coef[1]), "median_abs_rel_residual_by_band": {}}
for a, b in [(1, 2), (2, 5), (5, 20), (20, 60), (60, 300)]:
    m = (dd >= a) & (dd < b)
    v = float(np.median(np.abs(rel[m])))
    out["aim2_template_fit"]["median_abs_rel_residual_by_band"][f"{a}-{b} pitches"] = v
    print(f"   {a:3d}-{b:3d} pitches: median |residual| {v*100:.3f}% of the small-pad artifact")

print("\n5. Aim 3: thresholds of the uniform 1 um axon, 1 um above the chip (mV):")
offs = np.array([0, 5, 10, 15, 20]) * um
L, N = 1 * mm, 500
ax = Axon(n=N, dx_cm=L / N * 100, d_cm=1e-4, Ra_ohm_cm=150.0, dt_ms=0.001, membrane=MainenMembrane(8000.0))
steps = int(round(2.0 / 0.001)); sub = 20
tt = (np.arange(steps * sub) + 0.5) * (0.001 / sub) * 1e-3
out["aim3_thresholds_mV"] = {"offsets_um": (offs / um).tolist()}
for cname, (pad, C) in cases.items():
    iu = circuit_current(tt, biphasic(1.0, PHASE, 0.1e-3), res[pad]["R_solution_ohm"], C).reshape(steps, sub).mean(1)
    P = [straight_axon_points(np.array([sx, sy + o, 1 * um]), np.array([1.0, 0.0]), L, N)[0] for o in offs]
    Phi = np.column_stack([phi_box[pad](p) for p in P])
    th = thresholds(ax, Phi, iu, 2.0, [N // 2 - 150, N // 2 + 150], amp_max_V=3.0, iterations=9 if QUICK else 11) * 1e3
    out["aim3_thresholds_mV"][cname] = th.tolist()
    print(f"   {cname:31s} " + "  ".join(f"{o/um:3.0f}um {v:6.0f}" for o, v in zip(offs, th)))
(OUT / "summary.json").write_text(json.dumps(out, indent=2))
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/")
