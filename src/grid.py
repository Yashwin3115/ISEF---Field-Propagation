"""Nonuniform grid generation and size/memory checks.

Research plan step 4: 10 um spacing within 0.2 mm of the electrode tip,
growing by at most 10% per cell, up to 2 mm in the far field.
"""
import numpy as np


def one_side(D, h_min, r_fine, growth, h_max):
    """Node positions from 0 to D: spacing h_min out to r_fine, then growing
    by `growth` per cell up to h_max, then constant until D."""
    pos, h, pts = 0.0, h_min, [0.0]
    while pos + h < D and h < h_max:
        pos += h
        pts.append(pos)
        if pos >= r_fine - 1e-15:
            h = min(h * growth, h_max)
    # if the wall arrived mid-growth, don't leave a sliver cell against it
    if len(pts) > 1 and (D - pos) < 0.5 * (pts[-1] - pts[-2]):
        pts.pop()
        pos = pts[-1]
    # remaining stretch: equal cells no wider than the current spacing
    rest = D - pos
    n = max(1, int(np.ceil(rest / min(h, h_max) - 1e-9)))
    pts.extend(pos + rest * np.arange(1, n + 1) / n)
    return np.array(pts)


def make_axis(lo, hi, center, h_min, r_fine, growth, h_max):
    """Node coordinates on [lo, hi], finest around a node exactly at `center`.
    If center == lo, the grid is one-sided (used with a symmetry plane at lo)."""
    kw = dict(h_min=h_min, r_fine=r_fine, growth=growth, h_max=h_max)
    right = center + one_side(hi - center, **kw)
    if center <= lo:
        return right
    left = center - one_side(center - lo, **kw)[::-1]
    return np.concatenate([left, right[1:]])


def refine(x):
    """Halve every cell by inserting midpoints (keeps the grid's shape)."""
    mid = 0.5 * (x[:-1] + x[1:])
    out = np.empty(2 * len(x) - 1)
    out[0::2], out[1::2] = x, mid
    return out


# Peak memory of build + CG/AMG solve, bytes per unknown. Measured 756-872
# on the MacBook with the Oct 25 solver; the Nov 8 solver (full conductance
# matrix, floating electrodes) measured ~855-875 on Linux. Rounded up.
# Re-measure on the Mac with tests/measure_memory.py.
BYTES_PER_UNKNOWN = 1000


def size_report(x, y, z, n_unknowns=None, budget_gb=16.0, usable_fraction=0.75):
    """Print unknowns and estimated memory; raise if it will not fit.
    n_unknowns defaults to the interior nodes (all six faces fixed)."""
    m = n_unknowns if n_unknowns is not None else (len(x) - 2) * (len(y) - 2) * (len(z) - 2)
    est_gb = m * BYTES_PER_UNKNOWN / 1e9
    limit = budget_gb * usable_fraction
    hx, hy, hz = np.diff(x), np.diff(y), np.diff(z)
    growth = max(np.max(h[1:] / h[:-1]) for h in (hx, hy, hz))
    shrink = max(np.max(h[:-1] / h[1:]) for h in (hx, hy, hz))
    print(f"nodes per axis:   {len(x)} x {len(y)} x {len(z)}")
    print(f"unknowns:         {m:,}")
    print(f"spacing range:    {min(hx.min(), hy.min(), hz.min())*1e6:.1f} um "
          f"to {max(hx.max(), hy.max(), hz.max())*1e3:.2f} mm")
    print(f"max cell-to-cell growth: {max(growth, shrink):.3f}")
    print(f"estimated memory: {est_gb:.1f} GB (limit {limit:.0f} GB of {budget_gb:.0f})")
    if est_gb > limit:
        raise MemoryError("grid too large for available memory")
    return m, est_gb
