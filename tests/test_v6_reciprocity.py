"""V6: reciprocity.

Drive current I into electrode A and read electrode B (both measured against
the return). Then drive the same I into B and read A. Physics (and a
symmetric matrix) says the two readings must be identical, for ANY geometry
and conductivity -- so this test uses a deliberately lopsided setup:
  - two different-sized pads at unrelated positions
  - a return made of a distant pad plus the top face
  - a conductivity that changes with height (1.5 S/m below 20 um, 0.5 above)
The non-driven electrode floats (carries zero current), as a real
high-impedance recording electrode does.

Second part, for the unresolved recording electrodes used in production:
drive A and read the potential at a bare point P; then inject the same
current AT P and read A. These must match too.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grid import make_axis
from fd3d import face_mask, solve_general, node_volumes
from electrodes import square, stimulate

um = 1e-6
I = 1e-6
kw = dict(h_min=1 * um, r_fine=40 * um, growth=1.15, h_max=8 * um)
x = make_axis(-60 * um, 70 * um, 0, **kw)
y = make_axis(-50 * um, 60 * um, 0, **kw)
z = make_axis(0, 60 * um, 0, **kw)
shape = (len(x), len(y), len(z))
Z = np.broadcast_to(z, shape)
sigma = np.where(Z < 20 * um, 1.5, 0.5)

A = square(x, y, z, (-21 * um, -6 * um), 11.5 * um)
B = square(x, y, z, (17 * um, 11 * um), 8 * um)
ret = square(x, y, z, (45 * um, -35 * um), 15 * um) | face_mask(shape, ["z+"])
print(f"grid {shape}, {np.prod(shape):,} nodes; pads A {A.sum()} nodes, B {B.sum()} nodes\n")

rA = stimulate(x, y, z, sigma, A, "current", I, ret, recording_conductors=[B])
rB = stimulate(x, y, z, sigma, B, "current", I, ret, recording_conductors=[A])
V_B_from_A = float(np.mean(rA["phi_rel"][B]))
V_A_from_B = float(np.mean(rB["phi_rel"][A]))
d1 = abs(V_B_from_A / V_A_from_B - 1)
print("pad to pad")
print(f"  drive A, read B: {V_B_from_A*1e3:.6f} mV")
print(f"  drive B, read A: {V_A_from_B*1e3:.6f} mV   difference {d1:.1e}")

# bare point P on the substrate
iP, jP = np.argmin(abs(x - 40 * um)), np.argmin(abs(y - 30 * um))
V_P_from_A = float(rA["phi_rel"][iP, jP, 0])
src = np.zeros(shape)
src[iP, jP, 0] = I / node_volumes(x, y, z)[iP, jP, 0]
phi, st = solve_general(x, y, z, sigma, source=src, fixed=ret,
                        floating=[(A, 0.0), (B, 0.0)])
V_A_from_P = st["floating_potentials"][0]
d2 = abs(V_P_from_A / V_A_from_P - 1)
print("\npad to bare point")
print(f"  drive A, read P: {V_P_from_A*1e3:.6f} mV")
print(f"  drive P, read A: {V_A_from_P*1e3:.6f} mV   difference {d2:.1e}")

print("\nPASS" if max(d1, d2) < 1e-5 else "\nFAIL", "(criterion: identical to 1e-5)")
