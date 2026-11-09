"""V7: far field is the HALF-space formula.

Far from the electrode the disk looks like a point source on an insulating
plane, so phi -> I / (2 pi sigma r). The full-space formula I / (4 pi sigma r)
is half that. Check the ratio phi / (I / (2 pi sigma r)) along the axis and
along the substrate surface at r = 10a to 25a: it must be close to 1, not 0.5.
(At r = 10a the exact disk solution is still 0.3% below the point formula;
the comparison column shows that.)
"""
import numpy as np
from disk_setup import solve_disk, exact_phi, A_RADIUS as a, SIGMA as sigma0, um

I = 10e-6
r = solve_disk("current", I)
x, z, phi = r["x"], r["z"], r["phi"]
V0 = I / (4 * sigma0 * a)

print(f"{'path':>9} {'r/a':>5} {'phi/half-space':>15} {'exact disk/half-space':>22}")
worst = 0.0
for name, coords, line in [("axis", z, phi[0, 0, :]), ("substrate", x, phi[:, 0, 0])]:
    for k in [10, 15, 20, 25]:
        i = np.argmin(abs(coords - k * a))
        rr = coords[i]
        half = I / (2 * np.pi * sigma0 * rr)
        ex = exact_phi(0.0, rr, V0) if name == "axis" else exact_phi(rr, 0.0, V0)
        ratio = line[i] / half
        worst = max(worst, abs(ratio - ex / half))
        print(f"{name:>9} {rr/a:5.1f} {ratio:15.4f} {ex/half:22.4f}")

print(f"\nfull-space formula would give 0.5000. Largest gap to exact: {worst*100:.3f}%")
print("PASS" if worst < 0.02 else "FAIL")
