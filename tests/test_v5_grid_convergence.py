"""V5: grid convergence on the real electrode shape, with GCI.

An 11.5 um square electrode (the MaxOne pad) at 500 mV on the insulating
substrate, return = outer faces at 0 V, quarter symmetry. Three grids, each
made by halving every cell of the one before (constant refinement ratio r = 2).

For each quantity f (f1 finest, f3 coarsest):
    observed order   p = ln( (f3 - f2) / (f2 - f1) ) / ln 2
    extrapolated     f_ext = f1 + (f1 - f2) / (2^p - 1)        (Richardson)
    GCI (fine grid)  1.25 * |(f2 - f1) / f1| / (2^p - 1)       (Roache, 3 grids)

The current density is singular along the electrode edge, so p is expected
to fall below 2; the actual value is reported, not assumed.

Cross-check: the electrode can be drawn two ways on the grid -- including the
row of nodes ON the edge (pad slightly too big, by half a cell) or excluding
it (slightly too small). They converge from opposite sides. If both
extrapolate to the same current, the extrapolation is trustworthy.

Pass: every quantity converges monotonically, 0.5 < p < 3, and the two
drawings extrapolate to the same current within 1%.
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from grid import make_axis, refine
from fd3d import face_mask
from electrodes import square, stimulate, interpolate_at

um = 1e-6
SIDE = 11.5 * um                     # provisional MaxOne pad
PITCH = 17.5 * um
SIGMA = 1.5
V0 = 0.5
e = SIDE / 2
L = 20 * e                           # quarter box edge, 115 um

names = ["delivered current (uA)", "phi above centre, z = 11.5 um (mV)",
         "phi at next electrode, 17.5 um (mV)", "phi at 2nd electrode, 35 um (mV)"]
probe = np.array([[0, 0, SIDE], [PITCH, 0, 0], [2 * PITCH, 0, 0]])


def run(x, drop_edge):
    shape = (len(x),) * 3
    sigma = np.full(shape, SIGMA)
    ret = face_mask(shape, ["x+", "y+", "z+"])
    h = x[1] - x[0]
    stim = square(x, x, x, (0, 0), SIDE - h if drop_edge else SIDE)
    r = stimulate(x, x, x, sigma, stim, "voltage", V0, ret)
    I = 4 * r["delivered_current_A"] * 1e6
    v = interpolate_at(r["phi_rel"], x, x, x, probe) * 1e3
    return np.r_[I, v], r["solver"]["unknowns"]


def analyse(f):                       # f[0] coarsest ... f[2] finest
    f3, f2, f1 = f
    ratio = (f3 - f2) / (f2 - f1)
    monotone = ratio > 0
    p = np.log(np.abs(ratio)) / np.log(2)
    ext = f1 + (f1 - f2) / (2**p - 1)
    gci = 1.25 * np.abs((f2 - f1) / f1) / (2**p - 1)
    return monotone, p, ext, gci


x0 = make_axis(0, L, 0, h_min=e / 8, r_fine=e, growth=1.15, h_max=L / 6)
grids = [x0, refine(x0), refine(refine(x0))]

print(f"square pad {SIDE/um} um at {V0*1e3:.0f} mV, sigma {SIGMA} S/m, "
      f"outer faces grounded at {L/um:.0f} um (quarter)\n")
vals, vals_drop = [], []
for k, x in enumerate(grids):
    f, m = run(x, drop_edge=False)
    g, _ = run(x, drop_edge=True)
    vals.append(f); vals_drop.append(g)
    print(f"grid {k}: spacing at pad {(x[1]-x[0])/um:.3f} um, {m:,} unknowns, "
          f"I = {f[0]:.4f} uA (edge row dropped: {g[0]:.4f})", flush=True)

vals, vals_drop = np.array(vals), np.array(vals_drop)
mono, p, ext, gci = analyse(vals)
mono_d, p_d, ext_d, _ = analyse(vals_drop[:, :1])

print(f"\n{'quantity':<38} {'finest':>9} {'p':>5} {'extrapolated':>13} {'GCI':>7}")
for i, n in enumerate(names):
    print(f"{n:<38} {vals[2, i]:9.4f} {p[i]:5.2f} {ext[i]:13.4f} {gci[i]*100:6.2f}%"
          + ("" if mono[i] else "  NOT MONOTONE"))

agree = abs(ext[0] / ext_d[0] - 1)
print(f"\ncurrent, edge row included: -> {ext[0]:.4f} uA (p = {p[0]:.2f})")
print(f"current, edge row dropped:  -> {ext_d[0]:.4f} uA (p = {p_d[0]:.2f})")
print(f"the two drawings extrapolate to within {agree*100:.2f}% of each other")

passed = bool(np.all(mono) and mono_d[0] and np.all((p > 0.5) & (p < 3)) and agree < 0.01)
print("\nPASS" if passed else "\nFAIL")

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "v5_grid_convergence.json").write_text(json.dumps({
    "quantities": names, "values_coarse_to_fine": vals.tolist(),
    "observed_order_p": p.tolist(), "extrapolated": ext.tolist(),
    "gci_fine": gci.tolist(), "current_edge_dropped": vals_drop[:, 0].tolist(),
    "extrapolated_current_edge_dropped": float(ext_d[0]),
    "params": {"side_m": SIDE, "sigma": SIGMA, "V0": V0, "outer_box_m": L,
               "return": "outer faces x+, y+, z+ at 0 V", "reference": "return"},
    "passed": passed}, indent=2))
