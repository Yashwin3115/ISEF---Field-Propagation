"""V4: current conservation.

Voltage-mode disk. The net current flowing out through closed boxes of
increasing size around the electrode must equal the delivered current, and
the current arriving at the return must equal it with opposite sign.
Within 1%. (No current can leave through the substrate: it has no
neighbors below it, so there is no path -- the half-space check is V7.)
"""
import numpy as np
from disk_setup import solve_disk, A_RADIUS as a, QUARTER, um   # sets up the path to src/
from fd3d import current_out_of

r = solve_disk("voltage", 0.5)
x, y, z, L = r["x"], r["y"], r["z"], r["L"]
I = r["delivered_current_A"]
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

print(f"delivered current: {I*1e6:.4f} uA\n")
print(f"{'box half-size':>14} {'flux out (uA)':>14} {'error':>9}")
worst = 0.0
for s in [2, 5, 10, 20]:
    inside = (X <= s * a) & (Y <= s * a) & (Z <= s * a)
    flux = QUARTER * current_out_of(L, r["phi"], inside)
    e = abs(flux / I - 1)
    worst = max(worst, e)
    print(f"{s:>11} a  {flux*1e6:14.4f} {(flux/I-1)*100:8.4f}%")

e_ret = abs(-r["return_current_A"] / I - 1)
print(f"\nreturn contact: {r['return_current_A']*1e6:.4f} uA "
      f"(in = out within {e_ret*100:.4f}%)")
print("PASS" if max(worst, e_ret) < 0.01 else "FAIL", "(criterion < 1%)")
