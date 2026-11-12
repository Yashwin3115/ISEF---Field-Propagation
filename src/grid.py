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


def graded_axis(lo, hi, center, h_min, half_width, growth, h_far, zones=()):
    """Node coordinates on [lo, hi] for the MEA problem.

    - Spacing h_min across the stimulating pad, |s - center| <= half_width.
      half_width / h_min should be a whole number: then the pad edges fall
      exactly on nodes (they are placed at k * h_min, not added up step by step).
    - Outside the pad, spacing grows by `growth` per cell, but never exceeds
      the cap of the zone the node is in:
          zones = [(a, b, h_cap), ...]   spacing <= h_cap for a <= s <= b
      (e.g. the whole recording array, so every recording electrode is
      surrounded by fine-enough cells), and h_far everywhere else.
      Outside [a, b] the zone's cap does not jump straight to h_far; it rises
      gradually, h_cap + slope * (distance from the zone). The default slope,
      growth - 1, lets cells leaving a zone keep growing at the set rate. A 4th
      entry sets the slope; 1 - 1/growth makes cells approaching a zone shrink
      by at most the set rate (used around the return ring). Because the caps
      change smoothly, the wall-landing step below always works.
    """
    n_pad = int(round(half_width / h_min))
    if abs(n_pad * h_min - half_width) > 1e-9 * half_width:
        raise ValueError("half_width must be a whole number of h_min steps")

    def cap(s):
        c = h_far
        for zone in zones:
            a, b, hc = zone[:3]
            slope = zone[3] if len(zone) > 3 else growth - 1
            if a <= s <= b:
                c = min(c, hc)
            else:
                c = min(c, hc + slope * (a - s if s < a else s - b))
        return c

    def walk(direction, D, gg, n_cells=None):
        """Pad nodes, then cells growing by gg (capped by zones).
        n_cells None: stop before the first cell that would reach D, and also
        return that cell's size. Otherwise: exactly n_cells cells past the pad."""
        n0 = min(n_pad, int(np.floor(D / h_min + 1e-9)))
        pts = list(np.arange(n0 + 1) * h_min)            # exact pad nodes
        pos, h, count = pts[-1], h_min, 0
        while True:
            h = min(h * gg, cap(center + direction * pos))
            if n_cells is None and pos + h >= D:
                return np.array(pts), n0, h
            if n_cells is not None and count == n_cells:
                return np.array(pts), n0, h
            pos += h
            pts.append(pos)
            count += 1

    def march(direction, D):
        if D <= 0:
            return np.array([0.0])
        pts, n0, h_next = walk(direction, D, growth)
        d = np.diff(pts)
        rest = D - pts[-1]
        if len(d) <= n0:                     # wall reached inside the pad region
            return np.append(pts, D) if rest > 1e-12 * D else pts
        if rest <= 1e-12 * D:
            return pts
        h_last = d[-1]
        if rest >= h_last:
            # the leftover is at least as big as the last cell (and always
            # smaller than the next cell would have been): one more cell
            return np.append(pts, D)
        # Otherwise add one more cell and lower the growth rate slightly
        # (bisection) until the cells land exactly on the wall. The end
        # position rises smoothly with the growth rate, so this always works,
        # and growth never exceeds `growth` (zone caps are still respected).
        n_cells = len(d) - n0 + 1
        lo, hi_g = 1.0, growth
        if walk(direction, D, lo, n_cells)[0][-1] >= D:
            # only possible if the wall is a few cells past the pad
            return np.append(pts[:-1], D) if len(pts) > n0 + 1 else np.append(pts, D)
        for _ in range(200):
            mid = 0.5 * (lo + hi_g)
            if walk(direction, D, mid, n_cells)[0][-1] < D:
                lo = mid
            else:
                hi_g = mid
        out = walk(direction, D, hi_g, n_cells)[0]
        out[-1] = D                           # remove the last rounding error
        return out

    right = center + march(+1, hi - center)
    left = center - march(-1, center - lo)[::-1]
    out = np.concatenate([left, right[1:]])
    out[0], out[-1] = lo, hi
    return out


def refine(x):
    """Halve every cell by inserting midpoints (keeps the grid's shape)."""
    mid = 0.5 * (x[:-1] + x[1:])
    out = np.empty(2 * len(x) - 1)
    out[0::2], out[1::2] = x, mid
    return out


# Peak memory of build + CG/AMG solve, bytes per unknown. Measured 756-872
# on the MacBook with the Oct 25 solver; ~855-875 on Linux with the Nov 8
# solver; ~1015-1075 on Linux with classical multigrid (Nov 9). Rounded up.
# Re-measure on the Mac with tests/measure_memory.py.
BYTES_PER_UNKNOWN = 1150


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
