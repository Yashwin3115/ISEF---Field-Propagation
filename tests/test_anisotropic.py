"""Research plan Part 1, step 2: anisotropic conductivity diag(sx, sy, sz).

Exact solution for a point source I in an unbounded anisotropic medium:

    phi = I / (4 pi sqrt(sx sy sz)) * (x^2/sx + y^2/sy + z^2/sz)^(-1/2)

Same setup as step 7: one octant of a cube with the source at its corner
(the diagonal tensor keeps all three symmetry planes), exact potential on the
three outer faces. Ratios 1:1:1, 3:1:1 and 9:1:1 (sx : sy : sz, sy = sz =
1 S/m). Each is solved on three grids, every spacing (and the growth rate)
coarsened by 2 each time, and the error is measured at the same physical
points, 2-8 mm from the source in many directions.

Pass if, for every ratio:
  - max relative error on the finest grid < 1%
  - observed order of convergence 1.8-2.3 (second order, like the isotropic
    solver), from the error at the three grids
  - 1:1:1 matches the isotropic solver (one array) to round-off
"""
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fd3d import solve_3d, face_mask, cell_widths
from grid import make_axis
from scipy.interpolate import RegularGridInterpolator

mm = 1e-3
I = 50e-6
L = 20 * mm
RATIOS = [(1, 1, 1), (3, 1, 1), (9, 1, 1)]
FACTORS = [2.0, 1.0, 0.5]           # coarse -> fine; spacing x factor

def exact(X, Y, Z, s):
    sx, sy, sz = s
    with np.errstate(divide="ignore"):
        return I / (4 * np.pi * np.sqrt(sx * sy * sz)) / np.sqrt(X**2 / sx + Y**2 / sy + Z**2 / sz)

# fixed physical test points: 2-8 mm, 40 directions in the octant
rng = np.random.default_rng(3)
dirs = np.abs(rng.normal(size=(40, 3)))
dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
dirs = np.vstack([dirs, np.eye(3), np.ones((1, 3)) / np.sqrt(3)])
rr = np.array([2, 3, 4, 6, 8]) * mm
pts = (rr[:, None, None] * dirs[None, :, :]).reshape(-1, 3)

def run(s, f, iso_array=False):
    kw = dict(h_min=40e-6 * f, r_fine=0.4 * mm, growth=1 + 0.10 * f, h_max=1.0 * mm * f)
    x = make_axis(0, L, 0, **kw)
    y, z = x.copy(), x.copy()
    shape = (len(x), len(y), len(z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    ex = exact(X, Y, Z, s); ex[0, 0, 0] = 0.0
    del X, Y, Z
    fixed = face_mask(shape, ["x+", "y+", "z+"])
    src = np.zeros(shape)
    src[0, 0, 0] = (I / 8) / cell_widths(x)[0] ** 3
    if iso_array:
        sigma = np.full(shape, float(s[0]))
    else:
        sigma = tuple(np.full(shape, float(c)) for c in s)
    phi, st = solve_3d(x, y, z, sigma, src, ex, fixed)
    # potential x distance-like factor is smooth -> interpolate it, then divide
    rho = lambda P: np.sqrt(P[..., 0]**2 / s[0] + P[..., 1]**2 / s[1] + P[..., 2]**2 / s[2])
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")
    smooth = phi * rho(np.stack([Xg, Yg, Zg], -1))
    del Xg, Yg, Zg
    val = RegularGridInterpolator((x, y, z), smooth, method="cubic")(pts) / rho(pts)
    return val, len(x) ** 3

ok_all = True
out = {}
truth = None
for s in RATIOS:
    truth = exact(pts[:, 0], pts[:, 1], pts[:, 2], s)
    errs, sizes = [], []
    for f in FACTORS:
        v, n = run(s, f)
        errs.append(v / truth - 1)
        sizes.append(n)
    e = [np.max(np.abs(x)) for x in errs]
    # observed order from the three grids (ratio 2), on the error itself
    p_obs = np.log(np.sqrt(np.mean(errs[0]**2)) / np.sqrt(np.mean(errs[1]**2))) / np.log(2)
    p_obs2 = np.log(np.sqrt(np.mean(errs[1]**2)) / np.sqrt(np.mean(errs[2]**2))) / np.log(2)
    ok = e[-1] < 0.01 and 1.8 <= p_obs2 <= 2.3
    ok_all &= ok
    tag = ":".join(str(c) for c in s)
    out[tag] = {"max_rel_error_by_grid": e, "nodes": sizes, "order_coarse": p_obs, "order_fine": p_obs2}
    print(f"{'PASS' if ok else 'FAIL'}  {tag:6s} max error {e[0]*100:.3f}% -> {e[1]*100:.3f}% -> "
          f"{e[2]*100:.4f}%  observed order {p_obs:.2f}, {p_obs2:.2f}  (nodes {sizes})")

# isotropic consistency: tuple of three equal arrays == single array
v_t, _ = run((1, 1, 1), 2.0)
v_a, _ = run((1, 1, 1), 2.0, iso_array=True)
d = np.max(np.abs(v_t - v_a) / np.abs(v_a))
ok = d < 1e-10
ok_all &= ok
print(f"{'PASS' if ok else 'FAIL'}  (sx, sy, sz) = (1, 1, 1) vs one array: max difference {d:.1e}")
out["iso_consistency"] = float(d)

(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results" / "anisotropic_verification.json").write_text(json.dumps(out, indent=2))
print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
