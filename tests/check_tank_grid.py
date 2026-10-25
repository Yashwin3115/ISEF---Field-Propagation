"""Plan step 4: build the tank grids and record their size before any solve.
Working grid plus the two coarser comparison grids from step 10 (x4/3 and x2).
Tank 300 x 200 x 150 mm, electrode tip at the center.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grid import make_axis, size_report

mm = 1e-3
for factor in [1, 4 / 3, 2]:
    print(f"=== spacing x {factor:.3f} ===")
    kw = dict(h_min=10e-6 * factor, r_fine=0.2 * mm, growth=1.1, h_max=2 * mm * factor)
    x = make_axis(0, 300 * mm, 150 * mm, **kw)
    y = make_axis(0, 200 * mm, 100 * mm, **kw)
    z = make_axis(0, 150 * mm, 75 * mm, **kw)
    size_report(x, y, z)
    print()