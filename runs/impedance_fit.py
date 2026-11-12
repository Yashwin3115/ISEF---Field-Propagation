"""Electrode impedance sweep (plan step 56) -> Randles fit, and comparison of
the fitted solution resistance R_s with the solver's disk-to-return
resistance from the committed phantom predictions.

    python runs/impedance_fit.py data/phantom/impedance.csv 100000 \\
        [results/phantom_predictions/phantom-prediction-v1  sigma_S_per_m]

CSV columns: condition (cell | lifted), repeat, frequency_Hz, V_resistor,
V_electrode, phase_deg (electrode voltage relative to the current).
Amplitudes as read on the scope (peak-to-peak is fine if both are p-p).
"""
import sys
import json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scope import impedance_from_sweep, fit_randles
from phantom_analysis import read_csv

if len(sys.argv) < 3:
    sys.exit(__doc__)
rows = read_csv(sys.argv[1]); Rser = float(sys.argv[2])
def Z_of(cond):
    rs = [r for r in rows if r["condition"] == cond]
    fs = sorted(set(r["frequency_Hz"] for r in rs))
    Z = []
    for f in fs:
        zz = [impedance_from_sweep(r["frequency_Hz"], r["V_resistor"], r["V_electrode"],
                                   r["phase_deg"], Rser) for r in rs if r["frequency_Hz"] == f]
        Z.append(np.mean(zz))
    return np.array(fs), np.array(Z)
f, Z = Z_of("cell")
Zw = None
if any(r["condition"] == "lifted" for r in rows):
    fw, Zw_ = Z_of("lifted")
    Zw = np.interp(np.log(f), np.log(fw), Zw_.real) + 1j * np.interp(np.log(f), np.log(fw), Zw_.imag)
fit = fit_randles(f, Z, Zw)
for fi, zi in zip(f, Z):
    print(f"{fi:9.0f} Hz  |Z| {abs(zi):10.1f} ohm  phase {np.degrees(np.angle(zi)):6.1f} deg")
print(f"\nRandles fit: R_s {fit['R_s_ohm']:.1f} ohm, R_ct {fit['R_ct_ohm']:.3g} ohm, "
      f"CPE Q {fit['Q']:.3g}, alpha {fit['alpha']:.3f}; residual {fit['rms_log_residual']:.3f} (log|Z|), "
      f"{fit['rms_phase_residual_deg']:.1f} deg")
if len(sys.argv) >= 5:
    pred = json.loads((Path(sys.argv[3]) / "predictions.json").read_text())
    sig = float(sys.argv[4])
    Rpred = pred["part6"]["R_disk_to_return_x_sigma_by_grid"][-1] / sig
    print(f"solver: disk-to-return resistance through the saline {Rpred:.1f} ohm at {sig} S/m "
          f"-> R_s / prediction = {fit['R_s_ohm']/Rpred:.3f}")
