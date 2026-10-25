"""Test (d): point current source in a uniform medium.
Exact: phi(r) = I / (4 pi sigma r).
Values match the saline tank: I = 50 uA, sigma = 1.5 S/m (approx. 9 g/L NaCl),
electrode radius r0 = 0.25 mm, outer boundary R = 150 mm with the exact
potential applied there (as in research plan step 7).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd1d_radial import build_matrix_radial, build_rhs_radial, solve

I = 50e-6        # A
sigma0 = 1.5     # S/m
r0 = 0.25e-3     # m
R = 150e-3       # m


def phi_exact(r):
    return I / (4 * np.pi * sigma0 * r)


def run(N):
    r = np.linspace(r0, R, N + 1)
    A = build_matrix_radial(r, np.full(N + 1, sigma0))
    b = build_rhs_radial(r, np.zeros(N + 1), I, phi_exact(R))
    phi = solve(A, b)
    # relative error over 0.5-5 mm, the range in research plan step 7
    mask = (r >= 0.5e-3) & (r <= 5e-3)
    rel = np.abs(phi[mask] - phi_exact(r[mask])) / phi_exact(r[mask])
    return r[1] - r[0], rel.max()


print(f"{'N':>7} {'h (mm)':>9} {'max rel err':>12} {'ratio':>7}")
prev = None
ratios = []
for N in [600, 1200, 2400, 4800, 9600, 19200]:
    h, err = run(N)
    if prev:
        ratios.append(prev / err)
        print(f"{N:7d} {h*1e3:9.4f} {err:12.2e} {prev / err:7.2f}")
    else:
        print(f"{N:7d} {h*1e3:9.4f} {err:12.2e}       -")
    prev = err

order = np.log2(ratios[-1])
print(f"\nobserved order = {order:.2f} (expect 2)")
print(f"finest-grid error = {prev*100:.4f}% (plan requires < 2%)")
print("PASS" if abs(order - 2) < 0.1 and prev < 0.02 else "FAIL")