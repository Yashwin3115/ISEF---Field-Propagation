"""3D test: insulating (zero-current) faces, needed for symmetry planes (step 7)
and the insulating plane around a disk electrode (step 8).

Exact answer: phi = cos(pi x) cos(pi y) e^z on [0,1] x [0,0.8] x [0,0.6],
conductivity sigma = 1 + x + y + z. Its slope across the x = 0 and y = 0 faces
is zero, so those faces carry no current. They are left insulating (not fixed);
the other four faces are fixed to the exact potential. The solver has to find
the right values on the insulating faces by itself.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import build_system_3d, solve_3d, face_mask

Lx, Ly, Lz = 1.0, 0.8, 0.6
p = np.pi
FIXED_FACES = ["x+", "y+", "z-", "z+"]         # x- and y- are insulating


def phi_exact(X, Y, Z):
    return np.cos(p * X) * np.cos(p * Y) * np.exp(Z)


def sigma_of(X, Y, Z):
    return 1 + X + Y + Z


def source_of(X, Y, Z):
    cx, cy, sx, sy, ez = np.cos(p * X), np.cos(p * Y), np.sin(p * X), np.sin(p * Y), np.exp(Z)
    phi_x = -p * sx * cy * ez
    phi_y = -p * cx * sy * ez
    phi_z = cx * cy * ez
    lap = (1 - 2 * p**2) * cx * cy * ez
    return -(sigma_of(X, Y, Z) * lap + phi_x + phi_y + phi_z)


def grid(N):
    x = np.linspace(0, Lx, N + 1)
    y = np.linspace(0, Ly, N + 1)
    z = np.linspace(0, Lz, N + 1)
    return (x, y, z) + tuple(np.meshgrid(x, y, z, indexing="ij"))


x, y, z, X, Y, Z = grid(10)
fixed = face_mask(X.shape, FIXED_FACES)
A, _, _ = build_system_3d(x, y, z, sigma_of(X, Y, Z), source_of(X, Y, Z),
                          phi_exact(X, Y, Z), fixed)
asym = abs(A - A.T).max() / abs(A).max()
print(f"matrix symmetry: {asym:.1e}\n")

print(f"{'N':>4} {'unknowns':>9} {'CG iters':>8} {'max error':>11} "
      f"{'on insulating faces':>20} {'ratio':>6}")
prev, ratios = None, []
for N in [8, 16, 32, 64]:
    x, y, z, X, Y, Z = grid(N)
    fixed = face_mask(X.shape, FIXED_FACES)
    exact = phi_exact(X, Y, Z)
    phi, st = solve_3d(x, y, z, sigma_of(X, Y, Z), source_of(X, Y, Z), exact, fixed)
    err = np.max(np.abs(phi - exact))
    ins = face_mask(X.shape, ["x-", "y-"]) & ~fixed
    err_face = np.max(np.abs(phi - exact)[ins])
    r = f"{prev / err:6.2f}" if prev else "     -"
    if prev:
        ratios.append(prev / err)
    print(f"{N:4d} {st['unknowns']:9d} {st['iterations']:8d} {err:11.2e} "
          f"{err_face:20.2e} {r}")
    prev = err

order = np.log2(ratios[-1])
print(f"\nobserved order = {order:.2f} (expect 2)")
print("PASS" if abs(order - 2) < 0.15 and asym < 1e-12 else "FAIL")
