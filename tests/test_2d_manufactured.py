"""2D test: manufactured solution with variable conductivity.
Domain [0, 1] x [0, 0.5], so hx != hy (catches mixed-up spacings).
Exact answer: phi = sin(pi x) e^y,  conductivity: sigma = 1 + x + y.
Source from s = -(sigma * laplacian(phi) + dsigma/dx * dphi/dx + dsigma/dy * dphi/dy).
Edges get the exact phi, which is nonzero on two of them (tests boundary handling).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd2d import build_system_2d, solve_2d

Lx, Ly = 1.0, 0.5


def phi_exact(X, Y):
    return np.sin(np.pi * X) * np.exp(Y)


def sigma_of(X, Y):
    return 1 + X + Y


def source_of(X, Y):
    phi_x = np.pi * np.cos(np.pi * X) * np.exp(Y)
    phi_y = np.sin(np.pi * X) * np.exp(Y)
    lap = (1 - np.pi**2) * np.sin(np.pi * X) * np.exp(Y)
    return -(sigma_of(X, Y) * lap + phi_x + phi_y)


def run(N):
    x = np.linspace(0, Lx, N + 1)
    y = np.linspace(0, Ly, N + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    exact = phi_exact(X, Y)
    phi = solve_2d(x, y, sigma_of(X, Y), source_of(X, Y), exact)
    return x[1] - x[0], np.max(np.abs(phi - exact))


# symmetry check (needed for conjugate gradient later)
x = np.linspace(0, Lx, 11); y = np.linspace(0, Ly, 11)
X, Y = np.meshgrid(x, y, indexing="ij")
A, _ = build_system_2d(x, y, sigma_of(X, Y), source_of(X, Y), phi_exact(X, Y))
asym = abs(A - A.T).max()
print(f"matrix symmetry: max |A - A^T| = {asym:.1e}\n")

print(f"{'N':>5} {'unknowns':>9} {'max error':>12} {'ratio':>7}")
prev, ratios = None, []
for N in [10, 20, 40, 80, 160, 320]:
    h, err = run(N)
    r = f"{prev / err:7.2f}" if prev else "      -"
    if prev:
        ratios.append(prev / err)
    print(f"{N:5d} {(N - 1)**2:9d} {err:12.2e} {r}")
    prev = err

order = np.log2(ratios[-1])
print(f"\nobserved order = {order:.2f} (expect 2)")
print("PASS" if abs(order - 2) < 0.1 and asym < 1e-12 else "FAIL")