"""V1: disk electrode on an insulating plane, CURRENT mode.

The electrode is a floating metal disk carrying I. Compare the potential along
the axis above it with phi(z) = (I / (2 pi sigma a)) arctan(a / z), and the
electrode's own potential with I / (4 sigma a). Accept below 2%.
"""
import numpy as np
from disk_setup import solve_disk, A_RADIUS as a, SIGMA as sigma0, um

I = 10e-6                                  # A
r = solve_disk("current", I)
z = r["z"]

zz, ph = z[1:], r["phi"][0, 0, 1:]
exact = I / (2 * np.pi * sigma0 * a) * np.arctan(a / zz)
band = (zz >= 0.5 * a) & (zz <= 10 * a)
err_axis = np.max(np.abs(ph[band] / exact[band] - 1))

V_exact = I / (4 * sigma0 * a)
err_V = abs(r["stim_potential_V"] / V_exact - 1)

print(f"mode: {r['mode']}, I = {I*1e6:.1f} uA, a = {a/um:.1f} um, sigma = {sigma0} S/m")
print(f"unknowns {r['solver']['unknowns']:,}, CG iterations {r['solver']['iterations']}\n")
print(f"{'z (um)':>8} {'solver (mV)':>12} {'exact (mV)':>11} {'error':>8}")
for zq in np.array([0.5, 1, 2, 5, 10]) * a:
    k = np.argmin(abs(zz - zq))
    print(f"{zz[k]/um:8.2f} {ph[k]*1e3:12.3f} {exact[k]*1e3:11.3f} {(ph[k]/exact[k]-1)*100:7.3f}%")
print(f"\naxis, 0.5a to 10a: max error {err_axis*100:.3f}%")
print(f"electrode potential: {r['stim_potential_V']*1e3:.2f} mV vs exact {V_exact*1e3:.2f} mV "
      f"({err_V*100:+.3f}%)")
print("PASS" if max(err_axis, err_V) < 0.02 else "FAIL", "(criterion < 2%)")
