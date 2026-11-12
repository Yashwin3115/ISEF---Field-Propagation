"""Grid-convergence helpers shared by the run scripts (same procedure as
runs/predict_array.py, which keeps its own copy so the tagged v1 code is
unchanged)."""
import numpy as np

um, mm = 1e-6, 1e-3
LEVELS = [2.0, 2 ** 0.5, 1.0]          # coarse -> fine; every spacing x f


def grid_settings(f):
    return dict(n_pad=int(round(8 / f)), growth=1 + 0.10 * f,
                h_array=50 * um * f, h_far=0.5 * mm * f)


def gci_three(f1, f2, f3, N1, N2, N3):
    """Celik et al. (2008) procedure for three grids with any refinement ratios.
    f1 finest. Returns (p, extrapolated, GCI_fine (relative), monotone)."""
    h1, h2, h3 = N1 ** (-1 / 3), N2 ** (-1 / 3), N3 ** (-1 / 3)
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return np.nan, f1, 0.0, True
    s = np.sign(e32 / e21)
    monotone = s > 0
    p = 1.0
    for _ in range(100):
        q = np.log(abs((r21 ** p - s) / (r32 ** p - s)))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r21 ** p - 1)
    return p, ext, gci, monotone


def uncertainty_of(values_by_grid, N_by_grid):
    """values_by_grid: [coarse, medium, fine]. GCI if the three grids converge
    monotonically; otherwise the spread over the grids (relative), which is
    the honest fallback used for the predictions."""
    c, m, f = values_by_grid
    if not np.all(np.isfinite([c, m, f])):
        return np.nan, False
    p, ext, g, mono = gci_three(f, m, c, N_by_grid[2], N_by_grid[1], N_by_grid[0])
    spread = (max(c, m, f) - min(c, m, f)) / abs(f) if f else np.nan
    if mono and np.isfinite(p) and 0.5 <= p <= 4:
        return float(g), True
    return float(spread), False
