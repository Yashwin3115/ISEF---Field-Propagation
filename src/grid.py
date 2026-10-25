"""Nonuniform grid generation and size/memory checks.

Research plan step 4: 10 um spacing within 0.2 mm of the electrode tip,
growing by at most 10% per cell, up to 2 mm in the far field.
"""
import numpy as np


def make_axis(lo, hi, center, h_min, r_fine, growth, h_max):
    """Node coordinates on [lo, hi] with a node exactly at `center`.

    Spacing is h_min while within r_fine of center, then grows by `growth`
    per cell up to h_max, then stays at h_max until the wall.
    """
    def one_side(D):
        # fine region, then geometric growth until h reaches h_max
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

    left = center - one_side(center - lo)[::-1]
    right = center + one_side(hi - center)
    return np.concatenate([left, right[1:]])


def refine(x):
    """Halve every cell by inserting midpoints (keeps the grid's shape)."""
    mid = 0.5 * (x[:-1] + x[1:])
    out = np.empty(2 * len(x) - 1)
    out[0::2], out[1::2] = x, mid
    return out


# Peak memory of build + CG/AMG solve, bytes per unknown. Measured 756-872
# on the MacBook (tests/measure_memory.py, 1M-4M unknowns); rounded up.
BYTES_PER_UNKNOWN = 900


def size_report(x, y, z, budget_gb=16.0, usable_fraction=0.75):
    """Print unknowns and estimated memory; raise if it will not fit."""
    m = (len(x) - 2) * (len(y) - 2) * (len(z) - 2)
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