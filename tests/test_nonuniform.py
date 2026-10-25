"""Step 4 test: the solver on a nonuniform (stretched) grid.

Same manufactured solution as the uniform 3D test, but the grid is packed
tightly around an off-center point and grows by 25% per cell away from it,
which is harsher than the 10% used for the tank. The grid is then refined by
splitting every cell in half, twice. Second-order accuracy should survive:
error ratio ~4 per refinement.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import build_system_3d, solve_3d
from grid import make_axis, refine

Lx, Ly, Lz = 1.0, 0.8, 0.6
p = np.pi
c = (0.37, 0.41, 0.29)          # where the grid is finest (like an electrode tip)


def phi_exact(X, Y, Z):
    return np.sin(p * X) * np.sin(p * Y) * np.exp(Z)


def sigma_of(X, Y, Z):
    return 1 + X + Y + Z


def source_of(X, Y, Z):
    sx, sy, ez = np.sin(p * X), np.sin(p * Y), np.exp(Z)
    phi_x = p * np.cos(p * X) * sy * ez
    phi_y = p * sx * np.cos(p * Y) * ez
    phi_z = sx * sy * ez
    lap = (1 - 2 * p**2) * sx * sy * ez
    return -(sigma_of(X, Y, Z) * lap + phi_x + phi_y + phi_z)


def axis(lo, hi, center):
    return make_axis(lo, hi, center, h_min=0.005, r_fine=0.02, growth=1.25, h_max=0.08)


x0, y0, z0 = axis(0, Lx, c[0]), axis(0, Ly, c[1]), axis(0, Lz, c[2])
hs = [np.diff(a) for a in (x0, y0, z0)]
step = max(max(np.max(g[1:] / g[:-1]), np.max(g[:-1] / g[1:])) for g in hs)
print(f"base grid: spacing {min(g.min() for g in hs):.3f} to {max(g.max() for g in hs):.3f}, "
      f"neighbor-cell ratio up to {step:.2f}\n")

# symmetry check on the base grid
X, Y, Z = np.meshgrid(x0, y0, z0, indexing="ij")
A, _ = build_system_3d(x0, y0, z0, sigma_of(X, Y, Z), source_of(X, Y, Z), phi_exact(X, Y, Z))
asym = abs(A - A.T).max() / abs(A).max()
print(f"matrix symmetry: max |A - A^T| / max |A| = {asym:.1e}\n")

print(f"{'level':>5} {'unknowns':>9} {'CG iters':>8} {'max error':>11} {'ratio':>6}")
x, y, z = x0, y0, z0
prev, ratios = None, []
for level in range(3):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    exact = phi_exact(X, Y, Z)
    phi, st = solve_3d(x, y, z, sigma_of(X, Y, Z), source_of(X, Y, Z), exact)
    err = np.max(np.abs(phi - exact))
    r = f"{prev / err:6.2f}" if prev else "     -"
    if prev:
        ratios.append(prev / err)
    print(f"{level:5d} {st['unknowns']:9d} {st['iterations']:8d} {err:11.2e} {r}")
    prev = err
    x, y, z = refine(x), refine(y), refine(z)

order = np.log2(ratios[-1])
print(f"\nobserved order = {order:.2f} (expect 2)")
print("PASS" if abs(order - 2) < 0.15 and asym < 1e-12 else "FAIL")