"""Checks for the anisotropic-block model (src/aniso_phantom.py).

1. grid: on every refinement level each plate face and slot edge (all at
   half-millimetres) lies exactly midway between two nodes, in x and in y
2. unit cell, exact cases: a plate that is all slot conducts like plain
   saline scaled by nothing (sigma_y = sigma); aligned 1 mm slots every 2 mm
   in a 1 mm plate + 1 mm gap sit between the two exact bounds
3. unit cell converges with the grid (change between the two finest < 2%)
4. the three models in the tank on the coarsest grid: current conserved; the
   'tensor' model with (1, 1, 1) reproduces the isotropic run; a block makes
   readings along the plates larger than across them, an empty tank does not
5. bad inputs are refused (even slot width, odd pitch)
"""
import sys
import copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from phantom import load_setup, mm
from aniso_phantom import (block_axis, faces_offset, unit_cell_sigma_y, effective_tensor, solve_block,
                           aniso_geometry)

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

# 1
worst = []
for n in (1, 3, 5, 7):
    h = mm / n
    core = (1 * mm, 2 * mm)
    y = block_axis(0.10, 0.20, 0.10, 19.5 * mm, h, core, 1.12, 4 * mm, one_sided=True)
    x = block_axis(0.0, 0.30, 0.15, 19.5 * mm, h, core, 1.12, 4 * mm)
    worst.append(max(faces_offset((y - 0.10) / mm, 19.5), faces_offset((x - 0.15) / mm, 19.5)))
bad = faces_offset((block_axis(0.10, 0.20, 0.10, 19.5 * mm, mm / 3, (0.5 * mm, 1.5 * mm), 1.12, 4 * mm,
                               one_sided=True) - 0.10) / mm, 19.5)
check("half-millimetre boundaries midway between nodes", max(worst) < 1e-9 and bad > 0.05,
      f"worst offset on h = 1, 1/3, 1/5, 1/7 mm: {max(worst):.1e} mm; (a wrong core radius would give {bad:.3f} mm)")

# 2
full = unit_cell_sigma_y(2, slot_mm=3, staggered=False, n=3)          # slot wider than the cell: no plate left
al = unit_cell_sigma_y(2, slot_mm=1, staggered=False, n=9)
check("unit cell: exact and bounded cases", abs(full - 1) < 1e-6 and 0.5 < al < 1.0,
      f"all-slot plate {full:.6f} (exact 1); aligned 1 mm slots every 2 mm: {al:.4f} (between 0.5, the slot "
      f"column alone, and 1)")

# 3
v = [unit_cell_sigma_y(8, slot_mm=3, n=n) for n in (5, 9)]
T = effective_tensor(8, slot_mm=3, n=9)
check("unit cell converges", abs(v[1] / v[0] - 1) < 0.02 and abs(T[0] - 0.5) < 1e-12 and abs(T[2] - 0.6875) < 1e-12,
      f"sigma_y/sigma {v[0]:.4f} -> {v[1]:.4f}; tensor (x, y, z) = {T[0]:.3f}, {T[1]:.4f}, {T[2]:.4f} -> "
      f"{T[0]/T[1]:.2f}:1")

# 4
setup = load_setup(None)
var = next(iter(setup["anisotropic"]["variants"]))
iso = solve_block(setup, var, 1, "isotropic", verbose=False)
one = solve_block(setup, var, 1, "tensor", (1.0, 1.0, 1.0), verbose=False)
res = solve_block(setup, var, 1, "resolved", verbose=False)
an = setup["anisotropic"]
dx, dy = np.array(an["along_distances_mm"]), np.array(an["across_distances_mm"])
common = [d for d in dx if d in dy and d <= 10]
ratio = lambda r: np.mean([r["G_along"][list(dx).index(d)] / r["G_across"][list(dy).index(d)] for d in common])
check("tank runs: conservation, tensor(1,1,1) = isotropic, anisotropy seen",
      all(abs(r["current_check"] - 1) < 1e-6 for r in (iso, one, res))
      and np.allclose(one["G_along"], iso["G_along"], rtol=1e-6) and np.allclose(one["G_across"], iso["G_across"], rtol=1e-6)
      and abs(ratio(iso) - 1) < 0.05 and ratio(res) > 1.15,
      f"current {res['current_check']:.8f}; along/across (2-10 mm): empty tank {ratio(iso):.3f}, block {ratio(res):.2f}")

# 5
refused = 0
for key, val in (("slot_width_mm", 2.0), ("slot_pitch_mm", 7.0), ("plate_thickness_mm", 1.5)):
    s2 = copy.deepcopy(setup); s2["anisotropic"]["variants"][var][key] = val
    try:
        aniso_geometry(s2, var)
    except ValueError:
        refused += 1
check("bad block dimensions refused", refused == 3, f"{refused}/3")

print("ALL PASS" if ok_all else "SOME FAILED")
