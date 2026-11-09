"""Research plan step 7: point source at the center of a 300 mm saline cube.

The exact potential I/(4 pi sigma r) is applied on the outer faces, so cutting
the infinite medium down to a box adds no error. Pass if the relative error is
below 2% at every distance from 0.5 to 5 mm.

Symmetry: the source sits at the center of the cube and the medium is uniform,
so the potential is the same in all eight octants. We solve ONE octant
(0 to 150 mm on each axis) with the source at its corner. The three faces
through the source are symmetry planes, so no current crosses them; they are
left insulating. The corner node's control volume is 1/8 of a full one, so it
receives I/8. This gives the full-cube answer with 1/8 the unknowns: the
full cube at the plan's spacing would need ~22 million unknowns (~20 GB),
the octant needs ~2.8 million (~2.5 GB).

Usage:
    python tests/step7_point_source_cube.py          working grid
    python tests/step7_point_source_cube.py 1.333    all spacings x 4/3 (step 10)
    python tests/step7_point_source_cube.py 2        all spacings x 2   (step 10)
"""
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fd3d import solve_3d, face_mask, cell_widths
from grid import make_axis, size_report

factor = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
tag = {1.0: "x1"}.get(factor, f"x{factor:.3g}")

mm = 1e-3
I = 50e-6            # A
sigma0 = 1.5         # S/m
L = 150 * mm         # half the cube side

# --- grid (plan step 4), every spacing multiplied by `factor` ---
# In the growth zone each cell is ~10% wider than the one before, so there the
# spacing is about 10% of the distance from the source (h ~ 0.1 r). Multiplying
# h_min and h_max alone would leave that zone unchanged, so the growth rate
# is scaled too: 1.10 -> 1.133 (x4/3) -> 1.20 (x2). Then every cell is coarser.
growth = 1 + 0.10 * factor
kw = dict(h_min=10e-6 * factor, r_fine=0.2 * mm, growth=growth, h_max=2 * mm * factor)
x = make_axis(0, L, 0, **kw)           # one-sided: finest at the source, x = 0
y, z = x.copy(), x.copy()
shape = (len(x), len(y), len(z))
fixed = face_mask(shape, ["x+", "y+", "z+"])     # outer faces; x-, y-, z- are symmetry planes

print(f"=== Step 7, octant of 300 mm cube, spacing x {factor:.3g} ===")
size_report(x, y, z, n_unknowns=int((~fixed).sum()))

# --- exact solution, source, conductivity ---
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
R = np.sqrt(X**2 + Y**2 + Z**2)
with np.errstate(divide="ignore"):
    exact = I / (4 * np.pi * sigma0 * R)
exact[0, 0, 0] = 0.0                      # singular point, not used
del X, Y, Z

dx = cell_widths(x)
source = np.zeros(shape)
source[0, 0, 0] = (I / 8) / dx[0]**3      # this octant's share of the current

print("\nsolving (this can take several minutes)...")
phi, st = solve_3d(x, y, z, np.full(shape, sigma0), source, exact, fixed)
print(f"done: {st['iterations']} CG iterations, residual {st['rel_residual']:.1e}, "
      f"setup {st['setup_s']:.0f} s + solve {st['solve_s']:.0f} s")

# --- compare at every node from 0.5 to 5 mm ---
band = (R >= 0.5 * mm) & (R <= 5 * mm)
rel = (phi[band] - exact[band]) / exact[band]
worst = np.max(np.abs(rel))
print(f"\nnodes between 0.5 and 5 mm: {band.sum():,}")
print(f"max |relative error|: {worst*100:.3f}%   (mean {np.mean(np.abs(rel))*100:.3f}%)")

# --- table at the plan's distances along two directions ---
def along(line_phi, line_r, r_query):
    # phi * r is smooth (constant for 1/r), so interpolate that, then divide by r
    return np.interp(r_query, line_r, line_phi * line_r) / r_query

dists = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]) * mm
ax_r, ax_phi = x[1:], phi[1:, 0, 0]                       # along an axis
dg_r, dg_phi = np.sqrt(3) * x[1:], phi[np.arange(1, len(x)),
                                       np.arange(1, len(x)), np.arange(1, len(x))]  # diagonal
theory = I / (4 * np.pi * sigma0 * dists)
e_ax = along(ax_phi, ax_r, dists) / theory - 1
e_dg = along(dg_phi, dg_r, dists) / theory - 1

print(f"\n{'r (mm)':>7} {'exact (mV)':>11} {'axis err':>9} {'diagonal err':>13}")
for d, t, a, g in zip(dists, theory, e_ax, e_dg):
    print(f"{d/mm:7.1f} {t*1e3:11.4f} {a*100:8.3f}% {g*100:12.3f}%")

passed = worst < 0.02
print("\nPASS" if passed else "\nFAIL", "(criterion: < 2% at every distance from 0.5 to 5 mm)")

# --- save numbers and a figure ---
(ROOT / "results").mkdir(exist_ok=True)
out = {
    "step": 7, "grid_factor": factor, "unknowns": st["unknowns"],
    "cg_iterations": st["iterations"], "rel_residual": float(st["rel_residual"]),
    "max_rel_error_0.5_to_5mm": float(worst),
    "distances_mm": (dists / mm).tolist(),
    "phi_axis_mV": (along(ax_phi, ax_r, dists) * 1e3).tolist(),
    "phi_diag_mV": (along(dg_phi, dg_r, dists) * 1e3).tolist(),
    "phi_exact_mV": (theory * 1e3).tolist(),
    "passed": bool(passed),
}
(ROOT / "results" / f"step7_{tag}.json").write_text(json.dumps(out, indent=2))

mask = (ax_r >= 0.05 * mm) & (ax_r <= 20 * mm)
mask_d = (dg_r >= 0.05 * mm) & (dg_r <= 20 * mm)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.axvspan(0.5, 5, color="0.94", lw=0)
ax.axhline(2, color="0.5", lw=1, ls="--")
ax.axhline(-2, color="0.5", lw=1, ls="--")
ax.text(5.3, 2.1, "±2% limit", color="0.4", fontsize=8, va="bottom")
ax.text(np.sqrt(0.5 * 5), 3.0, "tested range\n0.5–5 mm", color="0.4", fontsize=8,
        ha="center", va="center")
ax.semilogx(ax_r[mask] / mm, (ax_phi[mask] / (I / (4*np.pi*sigma0*ax_r[mask])) - 1) * 100,
            color="#1f4e8c", lw=2, label="along an axis")
ax.semilogx(dg_r[mask_d] / mm, (dg_phi[mask_d] / (I / (4*np.pi*sigma0*dg_r[mask_d])) - 1) * 100,
            color="#c2571a", lw=2, label="along the diagonal")
ax.set_ylim(-3, 4.5)
ax.set_xlabel("distance from source (mm)")
ax.set_ylabel("error vs. I/(4πσr)  (%)")
ax.set_title(f"Step 7: point source in 300 mm cube (spacing ×{factor:.3g})")
ax.grid(True, which="both", color="0.92", lw=0.5)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"step7_error_{tag}.png", dpi=200)
print(f"saved results/step7_{tag}.json and figures/step7_error_{tag}.png")
