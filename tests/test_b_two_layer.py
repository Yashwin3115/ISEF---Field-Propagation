"""Test (b): two-layer slab, no source, phi(0)=0, phi(L)=1.
Layer 1 (sigma1) on [0, a], layer 2 (sigma2) on [a, L].
Exact: current J is the same in both layers, phi is piecewise linear.
The interface sits exactly halfway between two nodes.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd1d import build_matrix, build_rhs, solve

L = 1.0
sigma1, sigma2 = 1.0, 10.0
a = L / 2


def exact(x):
    J = 1.0 / (a / sigma1 + (L - a) / sigma2)
    return np.where(x <= a, J * x / sigma1, J * a / sigma1 + J * (x - a) / sigma2)


def run(N, averaging):
    x = np.linspace(0, L, N + 1)
    h = x[1] - x[0]
    sigma = np.where(x < a, sigma1, sigma2)
    A = build_matrix(sigma, averaging)
    b = build_rhs(np.zeros(N + 1), h, 0.0, 1.0)
    phi = solve(A, b)
    return h, np.max(np.abs(phi - exact(x)))


print(f"sigma1 = {sigma1}, sigma2 = {sigma2}, interface at x = {a}\n")
for averaging in ["arithmetic", "harmonic"]:
    print(f"--- {averaging} mean ---")
    print(f"{'N':>5} {'h':>10} {'max error':>12} {'ratio':>7}")
    prev = None
    for N in [11, 21, 41, 81, 161]:      # odd N puts the interface between nodes
        h, err = run(N, averaging)
        ratio = f"{prev / err:7.2f}" if prev and min(prev, err) > 1e-12 else "      -"
        print(f"{N:5d} {h:10.5f} {err:12.2e} {ratio}")
        prev = err
    print()

_, err_h = run(21, "harmonic")
print("PASS (harmonic exact)" if err_h < 1e-12 else "FAIL")