"""Full-array pipeline check against an exact solution.

Return = 'top_plate': insulating substrate at z = 0, fluid surface at z = H
held at 0 V, side walls far away. For a point current I on the substrate the
exact potential along the substrate is (method of images / eigenfunctions):

    phi(r) = I / (pi sigma H) * sum_{k>=0} K0( (2k+1) pi r / (2H) )

(K0 = modified Bessel function). Close to the source this tends to the
half-space I/(2 pi sigma r); farther out the grounded surface pulls it down.

The run uses the whole production pipeline -- graded grid over the 19 mm well,
MaxOne pad, floating current-driven pad, interpolation at all 26,400 electrode
centres -- on a deliberately coarse grid so it takes about a minute.
Pass: within 2% of the exact series at every electrode from 2 to 100 pitches
away, and current in = current out.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.special import k0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from maxone import transfer_run, PITCH, um, mm


def image_series(r, sigma, H, terms=20000):
    lam = (2 * np.arange(terms) + 1) * np.pi / (2 * H)
    return np.array([np.sum(k0(lam * q)) for q in np.atleast_1d(r)]) / (np.pi * sigma * H)


r = transfer_run({"return": "top_plate"}, n_pad=4, growth=1.2, h_array=100 * um)
p = r["params"]
d, Z = r["distance_m"], r["transfer_ohm"]
band = (d >= 2 * PITCH - 1e-12) & (d <= 100 * PITCH + 1e-12)
exact = image_series(d[band], p["sigma_S_per_m"], p["depth_m"])
err = np.abs(Z[band] / exact - 1)

print(f"\nunknowns {r['grid']['unknowns']:,}, CG iterations {r['solver']['iterations']}, "
      f"solve {r['solver']['setup_s'] + r['solver']['solve_s']:.0f} s")
print(f"current check {r['current_check']:.6f}, return check {r['return_check']:.6f}")
print(f"pad-to-return resistance {r['R_solution_ohm']/1e3:.2f} kOhm\n")
print(f"{'distance':>10} {'solver':>11} {'exact':>11} {'error':>8}")
for q in [2, 5, 10, 20, 50, 100]:
    i = np.argmin(np.abs(d - q * PITCH))
    ex = image_series(d[i], p["sigma_S_per_m"], p["depth_m"])[0]
    print(f"{d[i]/um:8.1f} um {Z[i]:9.2f} Ohm {ex:9.2f} Ohm {(Z[i]/ex-1)*100:+7.2f}%")
print(f"\n{band.sum():,} electrodes between 2 and 100 pitches: max error {err.max()*100:.2f}%, "
      f"mean {err.mean()*100:.2f}%")

ok = err.max() < 0.02 and abs(r["current_check"] - 1) < 1e-6 and abs(r["return_check"] - 1) < 1e-4
print("PASS" if ok else "FAIL")
