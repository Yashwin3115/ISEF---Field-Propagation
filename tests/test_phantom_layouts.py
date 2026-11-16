"""Checks of the phantom layouts on the solver (Nov 15 experiment list), on
the coarsest grid.

For every layout under "experiments" (flush in the plate, no plate, return =
top sheet, return = ring):
1. current conservation: what leaves the disk arrives at the return
2. RECIPROCITY on the solver: drive disk -> return and read probe - reference;
   then drive probe -> reference and read disk - return. The two transfer
   resistances must be equal (to solver precision) in EVERY layout, with the
   plate, post, sheet or ring in place. This is what the measured reciprocity
   check in runs/phantom_compare.py relies on.
3. the A - B difference is the same number at every recording position
Across layouts:
4. plate / no plate: about 2 from a few millimetres out (below 2 right next to
   the electrode, where the probe tip sits above the plane)
5. return position: ratio to the far-wall layout is ~1 next to the electrode
   and far from 1 at the farthest positions
6. conductivity collapse: solving with sigma = 0.8 and 1.55 S/m gives the same
   G = phi sigma / I
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from phantom import (load_setup, experiment_names, layout_setup, tank_geometry, tank_grid, tank_masks,
                     solve_tank, mm)
from fd3d import solve_general, current_out_of, node_volumes

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

setup = load_setup(None)
F = 4.0
names = experiment_names(setup)
runs = {}
cons, recip, flat = [], [], []
for name in names:
    s6 = layout_setup(setup, name)
    geo = tank_geometry(s6)
    r = solve_tank(setup, F, verbose=False, layout=name)
    runs[name] = (r, geo)
    cons.append(max(abs(r["current_check"] - 1), abs(r["return_check"] - 1)))
    flat.append(np.ptp(r["G_condition_A"] - r["G_condition_B"]) / abs(r["G_reference_wire"]))
    # reciprocity on the same grid and materials
    x, y, z = tank_grid(geo, F)
    shape = (len(x), len(y), len(z))
    plate, disk, ret, _ = tank_masks(geo, x, y, z)
    sigma = np.ones(shape); sigma[plate] = 1e-8
    I = 1.0
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape), floating=[(disk, I)], rtol=1e-11)
    free = ~(plate | disk | ret)
    zi = int(np.argmin(np.abs(z - (geo["pt"] + 1.0 * mm))))
    P = (int(np.argmin(np.abs(x - (geo["sx"] + 3 * mm)))), int(np.argmin(np.abs(y - (geo["sy"] + 4 * mm)))), zi)
    Q = (int(np.argmin(np.abs(x - 20 * mm))), len(y) - 3, int(np.argmin(np.abs(z - 50 * mm))))
    assert free[P] and free[Q], (name, "probe node is not in the saline")
    fwd = (phi[P] - phi[Q]) / I
    src = np.zeros(shape); vol = node_volumes(x, y, z)
    src[P] = I / vol[P]; src[Q] = -I / vol[Q]
    phi2, st2 = solve_general(x, y, z, sigma, source=src, fixed=ret, phi_fixed=np.zeros(shape),
                              floating=[(disk, 0.0)], rtol=1e-11)
    back = st2["floating_potentials"][0] / I             # disk minus return (return held at 0, carries no net current)
    leak = abs(current_out_of(st2["L"], phi2, ret)) / I
    recip.append((name, fwd, back, abs(back / fwd - 1), leak))
check("current conservation, every layout", max(cons) < 1e-6, f"worst mismatch {max(cons):.1e} ({len(names)} layouts)")
check("reciprocity on the solver, every layout", all(q[3] < 1e-6 and q[4] < 1e-6 for q in recip),
      "; ".join(f"{q[0]}: {q[1]:.6f} vs {q[2]:.6f} 1/m" for q in recip))
check("A - B is one number per layout", max(flat) < 1e-9, f"spread across positions {max(flat):.1e} of the value")

d = np.asarray(setup["part6"]["recording_distances_mm"], float)
GB = {n: runs[n][0]["G_condition_B"] for n in names}
if "E1_distance" in GB and "E2_no_plate" in GB:
    pr = GB["E1_distance"] / GB["E2_no_plate"]
    k = d <= 10
    check("plate / no plate", 1.5 < pr[0] < 2.2 and np.all(pr[k] > 1.3),
          "ratio (condition B) " + ", ".join(f"{a:g} mm {b:.2f}" for a, b in zip(d[k], pr[k])))
det, good = [], True
for n in names:
    if n.startswith("E3_"):
        rr = GB[n] / GB["E1_distance"]
        near, far = rr[0], rr[np.argmin(np.abs(d - 20))]
        good &= abs(near - 1) < 0.1 and abs(far - 1) > 0.15
        det.append(f"{n}: {near:.3f} at {d[0]:g} mm, {far:.2f} at 20 mm")
check("return position matters far away, not next to the electrode", good and det, "; ".join(det))

# 6 conductivity collapse (the solver scales exactly with sigma)
s6 = layout_setup(setup, names[0]); geo = tank_geometry(s6)
x, y, z = tank_grid(geo, F); shape = (len(x), len(y), len(z))
plate, disk, ret, _ = tank_masks(geo, x, y, z)
free = ~(plate | disk | ret)
Gs = []
for sg in (0.8, 1.55):
    sigma = np.full(shape, sg); sigma[plate] = 1e-8
    phi, _ = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape), floating=[(disk, 1.0)], rtol=1e-11)
    Gs.append(phi[free] * sg)
dev = float(np.max(np.abs(Gs[0] / np.where(Gs[1] == 0, 1, Gs[1]) - 1)[np.abs(Gs[1]) > 1e-3 * np.abs(Gs[1]).max()]))
check("two conductivities collapse (G = phi sigma / I)", dev < 1e-6, f"largest difference {dev:.1e}")

print("ALL PASS" if ok_all else "SOME FAILED")
