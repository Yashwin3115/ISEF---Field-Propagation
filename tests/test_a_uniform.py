"""Test (a): uniform slab, no source, phi(0)=0, phi(L)=1. Exact: phi = x/L."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd1d import build_matrix, build_rhs, solve

L = 1.0
N = 20                       # number of intervals
x = np.linspace(0, L, N + 1)
h = x[1] - x[0]

sigma = np.ones(N + 1)
source = np.zeros(N + 1)

A = build_matrix(sigma)
b = build_rhs(source, h, phi_left=0.0, phi_right=1.0)
phi = solve(A, b)

exact = x / L
err = np.max(np.abs(phi - exact))
print(f"max error = {err:.2e}")
print("PASS" if err < 1e-12 else "FAIL")