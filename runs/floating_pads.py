"""Spec 2.6: do the recording electrodes disturb the field?

The production model ignores the 26,399 recording pads: they draw no current
(high-impedance amplifiers), so their potential is read by interpolation at
their centres. But each pad is a patch of metal, which forces the fluid
touching it to one potential. This run brackets the truth with two extremes:

  A. pads ignored             -- metal fully hidden behind its interface
  B. pads = floating metal    -- metal in perfect contact with the fluid
                                 (equipotential, zero net current)

A real Pt-black pad sits in between: during the 7-15 us current spike its
interface impedance (roughly 450 kOhm x 1 kHz / 16 kHz ~ 30 kOhm) is similar
to the fluid's own resistance around a pad (~27 kOhm). If A and B agree,
the assumption is safe whatever the interface does.

Setup: the provisional settings (MaxOne, top-plate return, 1.5 S/m), current
mode, 1 uA into pad (110, 60). The 8 pads in the 3 x 3 block around it are
resolved exactly: the grid there is uniform (1.09375 um in x, 0.5 um in y), so
every pad edge falls on a node (pitch 17.5 = 16 x 1.09375 = 35 x 0.5;
half-widths 4.375 = 4 x 1.09375 and 6.0 = 12 x 0.5).

    python runs/floating_pads.py
"""
import sys
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from grid import graded_axis, size_report
from maxone import (CHIPS, PROVISIONAL, PITCH, electrode_xy, array_extent,
                    rectangle_on_substrate, return_mask, um, mm)
from electrodes import stimulate, interpolate_at
from fd3d import current_out_of

OUT = ROOT / "results" / "runs" / "floating_pads"
OUT.mkdir(parents=True, exist_ok=True)

p = dict(PROVISIONAL)
chip = CHIPS[p["chip"]]
c0, r0 = p["stim_col_row"]
sx, sy = electrode_xy(c0, r0)
px, py = chip["pad_x"] / 2, chip["pad_y"] / 2
HX, HY = px / 4, 0.5 * um                      # 1.09375 um, 0.5 um
assert abs(PITCH / HX - round(PITCH / HX)) < 1e-9 and abs(PITCH / HY - round(PITCH / HY)) < 1e-9

BLOCK = 1                                       # pads within 1 column / row
# coarser far away than the production grid (both runs share this grid,
# so the comparison is fair) to keep memory near 4 GB
growth, h_array, h_far = 1.10, 100 * um, 1.0 * mm
ax, ay = array_extent()
bx = (BLOCK + 0.5) * PITCH
x = graded_axis(-p["well_half_width_m"], p["well_half_width_m"], sx, HX, px, growth, h_far,
                zones=[(sx - bx, sx + bx, HX), (-ax - 0.2 * mm, ax + 0.2 * mm, h_array)])
y = graded_axis(-p["well_half_width_m"], p["well_half_width_m"], sy, HY, py, growth, h_far,
                zones=[(sy - bx, sy + bx, HY), (-ay - 0.2 * mm, ay + 0.2 * mm, h_array)])
z = graded_axis(0.0, p["depth_m"], 0.0, HY, 8 * HY, growth, h_far)
shape = (len(x), len(y), len(z))
print("floating-pad check: grid")
size_report(x, y, z)

stim = rectangle_on_substrate(x, y, z, (sx, sy), 2 * px, 2 * py)
neighbours = []
for dc in range(-BLOCK, BLOCK + 1):
    for dr in range(-BLOCK, BLOCK + 1):
        if dc == 0 and dr == 0:
            continue
        cx, cy = electrode_xy(c0 + dc, r0 + dr)
        m = rectangle_on_substrate(x, y, z, (cx, cy), 2 * px, 2 * py)
        ix = np.flatnonzero(m[:, :, 0].any(axis=1))
        iy = np.flatnonzero(m[:, :, 0].any(axis=0))
        assert abs(x[ix[-1]] - x[ix[0]] - 2 * px) < 1e-12 and abs(y[iy[-1]] - y[iy[0]] - 2 * py) < 1e-12
        neighbours.append(((dc, dr), (cx, cy), m))
ret = return_mask(p["return"], x, y, z)
sigma = np.full(shape, p["sigma_S_per_m"])
I = 1e-6

runs = {}
for name, cond in [("A_ignored", []), ("B_floating_metal", [m for _, _, m in neighbours])]:
    t0 = time.perf_counter()
    r = stimulate(x, y, z, sigma, stim, "current", I, ret, recording_conductors=cond, params=p)
    print(f"{name}: {r['solver']['iterations']} CG iterations, {time.perf_counter()-t0:.0f} s, "
          f"R(pad->return) {r['stim_potential_V']/I/1e3:.3f} kOhm")
    runs[name] = r

# potential each neighbour reports, per uA
#   A: the fluid's potential averaged over the pad's footprint (what a pad would
#      read if it did not disturb anything), and at its centre point (what the
#      production model uses)
#   B: the floating pad's own potential
rows = []
for (dc, dr), (cx, cy), m in neighbours:
    A = runs["A_ignored"]["phi_rel"]
    B = runs["B_floating_metal"]["phi_rel"]
    a_centre = float(interpolate_at(A, x, y, z, np.array([cx, cy, 0.0]))[0])
    a_mean = float(A[m].mean())
    b_pad = float(B[m].mean())
    rows.append({"dcol": dc, "drow": dr, "distance_um": float(np.hypot(cx - sx, cy - sy) / um),
                 "A_centre_mV_per_uA": a_centre / I * 1e-3, "A_footprint_mV_per_uA": a_mean / I * 1e-3,
                 "B_floating_mV_per_uA": b_pad / I * 1e-3,
                 "B_vs_A_centre_percent": 100 * (b_pad / a_centre - 1),
                 "footprint_vs_centre_percent": 100 * (a_mean / a_centre - 1)})

# farther electrodes (not resolved in either run): interpolated
far = []
for k in [5, 10, 20, 50, 100]:
    cx, cy = electrode_xy(c0 + k, r0)
    va = float(interpolate_at(runs["A_ignored"]["phi_rel"], x, y, z, np.array([cx, cy, 0.0]))[0])
    vb = float(interpolate_at(runs["B_floating_metal"]["phi_rel"], x, y, z, np.array([cx, cy, 0.0]))[0])
    far.append({"pitches": k, "A_mV_per_uA": va / I * 1e-3, "B_mV_per_uA": vb / I * 1e-3,
                "B_vs_A_percent": 100 * (vb / va - 1)})

RA = runs["A_ignored"]["stim_potential_V"] / I
RB = runs["B_floating_metal"]["stim_potential_V"] / I

print(f"\n{'pad':>9} {'dist um':>8} {'A centre':>9} {'A footprint':>12} {'B metal':>9} "
      f"{'B vs A':>8} {'footprint vs centre':>20}   (mV per uA)")
for q in sorted(rows, key=lambda q: (q["distance_um"], q["dcol"], q["drow"])):
    if q["dcol"] < 0 or q["drow"] < 0:
        continue                                 # symmetric partners, skip for brevity
    print(f"({q['dcol']:+d},{q['drow']:+d}) {q['distance_um']:8.1f} {q['A_centre_mV_per_uA']:9.4f} "
          f"{q['A_footprint_mV_per_uA']:12.4f} {q['B_floating_mV_per_uA']:9.4f} "
          f"{q['B_vs_A_centre_percent']:+7.2f}% {q['footprint_vs_centre_percent']:+19.2f}%")
print(f"\n{'pitches':>7} {'A':>9} {'B':>9} {'B vs A':>8}   (beyond the resolved block)")
for q in far:
    print(f"{q['pitches']:7d} {q['A_mV_per_uA']:9.4f} {q['B_mV_per_uA']:9.4f} {q['B_vs_A_percent']:+7.2f}%")
print(f"\nstimulating pad to return: A {RA/1e3:.3f} kOhm, B {RB/1e3:.3f} kOhm "
      f"({100*(RB/RA-1):+.2f}%)")
for name, r in runs.items():
    print(f"current check {name}: in {r['delivered_current_A']/I:.6f}, "
          f"out at return {-r['return_current_A']/I:.6f}")

worst_near = max(abs(q["B_vs_A_centre_percent"]) for q in rows)
worst_far = max(abs(q["B_vs_A_percent"]) for q in far)
print(f"\nlargest effect of treating the neighbours as metal: {worst_near:.2f}% on the "
      f"neighbours themselves, {worst_far:.2f}% at 5-100 pitches")

(OUT / "summary.json").write_text(json.dumps({
    "settings": p, "grid_nodes": list(shape), "block": f"{2*BLOCK+1} x {2*BLOCK+1}",
    "neighbours": rows, "farther": far,
    "R_pad_to_return_ohm": {"A_ignored": RA, "B_floating_metal": RB},
    "largest_effect_neighbours_percent": worst_near,
    "largest_effect_far_percent": worst_far}, indent=2, default=float))
print(f"wrote {OUT.relative_to(ROOT)}/summary.json")
