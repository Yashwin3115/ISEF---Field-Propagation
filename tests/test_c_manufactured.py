"""Test (c): manufactured solution with variable conductivity.
Chosen exact answer: phi(x) = sin(pi x) on [0, 1], with sigma(x) = 1 + x.
Plugging into d/dx(sigma dphi/dx) = -s gives the source
    s(x) = (1 + x) pi^2 sin(pi x) - pi cos(pi x).
The solver should recover phi with error shrinking like h^2 (ratio ~4).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd1d import build_matrix, build_rhs, solve

L = 1.0


def phi_exact(x):
    return np.sin(np.pi * x)


def sigma_of(x):
    return 1.0 + x


def source_of(x):
    return (1 + x) * np.pi**2 * np.sin(np.pi * x) - np.pi * np.cos(np.pi * x)


def run(N):
    x = np.linspace(0, L, N + 1)
    h = x[1] - x[0]
    A = build_matrix(sigma_of(x), "harmonic")
    b = build_rhs(source_of(x), h, phi_exact(x[0]), phi_exact(x[-1]))
    phi = solve(A, b)
    return h, np.max(np.abs(phi - phi_exact(x)))


print(f"{'N':>5} {'h':>10} {'max error':>12} {'ratio':>7}")
prev = None
ratios = []
for N in [10, 20, 40, 80, 160, 320]:
    h, err = run(N)
    if prev:
        ratios.append(prev / err)
        print(f"{N:5d} {h:10.5f} {err:12.2e} {prev / err:7.2f}")
    else:
        print(f"{N:5d} {h:10.5f} {err:12.2e}       -")
    prev = err

order = np.log2(ratios[-1])
print(f"\nobserved order of accuracy = {order:.2f} (expect 2)")
print("PASS" if abs(order - 2) < 0.1 else "FAIL")