"""MaxOne layout and grid checks (no solve, runs in a second or two).

  - 26,400 electrodes on a 220 x 120 grid at 17.5 um pitch
  - array covers 3.85 x 2.10 mm (User Guide)
  - MaxOne pad area 8.75 x 12.00 = 105 um^2 (User Guide table)
  - on the solver grid, the stimulating pad is drawn with its edges exactly on
    nodes, for both chip types, at several stimulation positions
  - in x and y, spacing never grows by more than the set rate (plus up to 1%
    where the last stretch is re-split to end exactly on the well wall), and is
    never coarser than the cap over the array
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from maxone import (CHIPS, PITCH, N_COLS, N_ROWS, all_electrodes, electrode_xy,
                    array_extent, build_grid, rectangle_on_substrate, um, mm)

ok = []
col, row, x, y = all_electrodes()
n = len(col)
dx = np.unique(np.round(np.diff(np.unique(x)) / um, 6))
dy = np.unique(np.round(np.diff(np.unique(y)) / um, 6))
ax, ay = array_extent()
ok += [n == 26400, list(dx) == [17.5], list(dy) == [17.5],
       abs(2 * ax / mm - 3.85) < 1e-9, abs(2 * ay / mm - 2.10) < 1e-9]
print(f"electrodes {n:,}; pitch {dx} x {dy} um; array {2*ax/mm:.2f} x {2*ay/mm:.2f} mm")

a = CHIPS["MaxOne"]["pad_x"] * CHIPS["MaxOne"]["pad_y"] / um**2
ok.append(abs(a - 105) < 1e-9)
print(f"MaxOne pad area {a:.2f} um^2 (User Guide: 105)")

for chip in CHIPS:
    for cr in [(110, 60), (0, 0), (219, 119), (5, 60)]:
        sx, sy = electrode_xy(*cr)
        X, Y, Z = build_grid(chip, (sx, sy), 3.5 * mm, 9.5 * mm, n_pad=4, growth=1.2,
                             h_array=100 * um)
        px, py = CHIPS[chip]["pad_x"] / 2, CHIPS[chip]["pad_y"] / 2
        edges = [np.min(np.abs(X - (sx + s * px))) for s in (-1, 1)] + \
                [np.min(np.abs(Y - (sy + s * py))) for s in (-1, 1)]
        m = rectangle_on_substrate(X, Y, Z, (sx, sy), 2 * px, 2 * py)
        ix = np.flatnonzero(m[:, :, 0].any(axis=1))
        iy = np.flatnonzero(m[:, :, 0].any(axis=0))
        drawn = (X[ix[-1]] - X[ix[0]], Y[iy[-1]] - Y[iy[0]])
        g = max(np.max(h[1:] / h[:-1]) for h in (np.diff(X), np.diff(Y)))
        cap = max(np.diff(X)[(X[:-1] > -ax) & (X[1:] < ax)].max(),
                  np.diff(Y)[(Y[:-1] > -ay) & (Y[1:] < ay)].max())
        good = (max(edges) < 1e-12 and abs(drawn[0] - 2 * px) < 1e-12
                and abs(drawn[1] - 2 * py) < 1e-12 and g <= 1.2 * 1.01 and cap <= 100 * um * (1 + 1e-9))
        ok.append(good)
        print(f"{chip:8s} pad {cr}: edges on nodes {max(edges)/um:.1e} um, drawn "
              f"{drawn[0]/um:.2f} x {drawn[1]/um:.2f} um, max growth {g:.3f}, "
              f"largest cell over array {cap/um:.0f} um {'ok' if good else 'BAD'}")

print("PASS" if all(ok) else "FAIL")
