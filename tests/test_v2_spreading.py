"""V2: spreading resistance, VOLTAGE mode.

Hold the disk at V0 on the insulating plane. A disk on a half space has
spreading resistance R = 1/(4 sigma a), so the delivered current must be
I = 4 sigma a V0. The current is computed by summing the current leaving the
electrode's nodes (it is an output, not an input). Accept below 2%.

Independent arithmetic check of the spec's number:
    4 * 1.5 S/m * 6.5e-6 m * 0.5 V = 1.95e-5 A = 19.5 uA.
NOTE: this assumes all of V0 drops across the saline. A real electrode's
interface impedance takes much of it, so 19.5 uA is a solver test, not a
prediction of what the chip delivers.
"""
import numpy as np
from disk_setup import solve_disk, A_RADIUS as a, SIGMA as sigma0, um

V0 = 0.5
r = solve_disk("voltage", V0)
I = r["delivered_current_A"]
I_exact = 4 * sigma0 * a * V0
err = abs(I / I_exact - 1)

print(f"mode: {r['mode']}, V0 = {V0*1e3:.0f} mV, a = {a/um:.1f} um, sigma = {sigma0} S/m")
print(f"delivered current: {I*1e6:.3f} uA   exact 4*sigma*a*V0: {I_exact*1e6:.3f} uA   "
      f"error {(I/I_exact-1)*100:+.3f}%")
print(f"spreading resistance: {V0/I/1e3:.2f} kOhm   exact 1/(4 sigma a): "
      f"{1/(4*sigma0*a)/1e3:.2f} kOhm")
print("PASS" if err < 0.02 else "FAIL", "(criterion < 2%)")
