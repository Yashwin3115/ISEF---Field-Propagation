"""3D test: manufactured solution with variable conductivity, solved with CG + AMG.
Domain [0, 1] x [0, 0.8] x [0, 0.6], so all three spacings differ.
Exact answer: phi = sin(pi x) sin(pi y) e^z,  conductivity: sigma = 1 + x + y + z.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import build_system_3d, solve_cg_amg, solve_3d

Lx, Ly, Lz = 1.0, 0.8, 0.6
p = np.pi


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


def grid(N):
    x = np.linspace(0, Lx, N + 1)
    y = np.linspace(0, Ly, N + 1)
    z = np.linspace(0, Lz, N + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return x, y, z, X, Y, Z


# 1) matrix symmetry, and CG+AMG agrees with the direct solver
x, y, z, X, Y, Z = grid(12)
A, b = build_system_3d(x, y, z, sigma_of(X, Y, Z), source_of(X, Y, Z), phi_exact(X, Y, Z))
asym = abs(A - A.T).max()
u_cg, _ = solve_cg_amg(A, b)
u_direct = spsolve(A, b)
cg_vs_direct = np.max(np.abs(u_cg - u_direct)) / np.max(np.abs(u_direct))
print(f"matrix symmetry:  max |A - A^T| = {asym:.1e}")
print(f"CG+AMG vs direct: max rel. difference = {cg_vs_direct:.1e}\n")

# 2) convergence
print(f"{'N':>4} {'unknowns':>9} {'CG iters':>8} {'solve s':>8} {'max error':>11} {'ratio':>6}")
prev, ratios = None, []
for N in [8, 16, 32, 64]:
    x, y, z, X, Y, Z = grid(N)
    exact = phi_exact(X, Y, Z)
    phi, st = solve_3d(x, y, z, sigma_of(X, Y, Z), source_of(X, Y, Z), exact)
    err = np.max(np.abs(phi - exact))
    r = f"{prev / err:6.2f}" if prev else "     -"
    if prev:
        ratios.append(prev / err)
    print(f"{N:4d} {st['unknowns']:9d} {st['iterations']:8d} "
          f"{st['setup_s'] + st['solve_s']:8.2f} {err:11.2e} {r}")
    prev = err

order = np.log2(ratios[-1])
print(f"\nobserved order = {order:.2f} (expect 2)")
ok = abs(order - 2) < 0.1 and asym < 1e-12 and cg_vs_direct < 1e-6
print("PASS" if ok else "FAIL")