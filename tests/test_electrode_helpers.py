"""Checks on the electrode toolkit (src/electrodes.py):
  - interpolation at recording-electrode coordinates is exact for a linear field
  - reference-relative output: changing the reference shifts every value by a constant
  - a passive floating electrode carries zero net current and sits at one potential
  - a square electrode aligned with the grid has exactly its true area
  - bad inputs are refused (unknown mode, stimulating electrode overlapping the return)
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import face_mask, current_out_of
from electrodes import square, disk, interpolate_at, stimulate, patch_area

um = 1e-6
ok = []

# 1. interpolation: phi = 2x - 3y + 5z + 1 must be reproduced exactly
x = np.sort(np.r_[0, np.cumsum(np.linspace(1, 3, 12))]) * um
y, z = x.copy(), x[:8].copy()
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
lin = 2 * X - 3 * Y + 5 * Z + 1
pts = np.array([[3.3, 7.1, 2.2], [10.0, 1.5, 0.0], [20.2, 20.2, 9.9]]) * um
err = np.max(np.abs(interpolate_at(lin, x, y, z, pts) -
                    (2 * pts[:, 0] - 3 * pts[:, 1] + 5 * pts[:, 2] + 1)))
ok.append(err < 1e-12)
print(f"interpolation of a linear field: max error {err:.1e}")

# 2-3. grid with a stimulating square, a passive floating square, ground on the far faces
h = 0.5 * um
x = np.arange(0, 60 * um + h / 2, h)
y = np.arange(0, 40 * um + h / 2, h)
z = np.arange(0, 30 * um + h / 2, h)
shape = (len(x), len(y), len(z))
sigma = np.full(shape, 1.5)
stim = square(x, y, z, (15 * um, 20 * um), 11.5 * um)
rec = square(x, y, z, (40 * um, 20 * um), 11.5 * um)
ret = face_mask(shape, ["x+", "z+"])

r1 = stimulate(x, y, z, sigma, stim, "voltage", 0.5, ret, recording_conductors=[rec])
L = r1["L"]
I_rec = current_out_of(L, r1["phi"], rec)
spread = np.ptp(r1["phi"][rec])
ok.append(abs(I_rec) < 1e-6 * abs(r1["delivered_current_A"]) and spread < 1e-9)
print(f"floating recording electrode: net current {I_rec:.1e} A "
      f"(delivered {r1['delivered_current_A']*1e6:.3f} uA), potential spread {spread:.1e} V")

ref_pt = (55 * um, 5 * um, 25 * um)
r2 = stimulate(x, y, z, sigma, stim, "voltage", 0.5, ret, recording_conductors=[rec],
               reference=("point", ref_pt))
shift = r1["phi_rel"] - r2["phi_rel"]
ok.append(np.ptp(shift) < 1e-9 and abs(shift.mean() - r2["phi_ref"]) < 1e-9)
print(f"reference moved to a point: every value shifts by the same "
      f"{r2['phi_ref']*1e3:.4f} mV (spread {np.ptp(shift):.1e})")

# 4. square area on an aligned grid
area = patch_area(x, y, z, stim)
ok.append(abs(area / (11.5 * um)**2 - 1) < 1e-9)
print(f"square electrode area: {area/um**2:.4f} um^2 (true {11.5**2:.4f})")

# 5. refused inputs
for label, call in [
    ("unknown mode", lambda: stimulate(x, y, z, sigma, stim, "amps", 1e-6, ret)),
    ("stim overlaps return", lambda: stimulate(x, y, z, sigma, stim, "voltage", 0.5, ret | stim)),
]:
    try:
        call()
        ok.append(False); print(f"MISSED {label}")
    except ValueError as e:
        ok.append(True); print(f"refused {label}: {e}")

print("PASS" if all(ok) else "FAIL")
