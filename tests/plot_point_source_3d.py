"""First 3D picture: potential around a point electrode in saline.

Small 40 mm cube of saline, 50 uA injected at the center node, exact
potential I/(4 pi sigma r) on the walls. Solves on a nonuniform grid
(~1M unknowns, about a minute), then draws:
  1. a 3D surface of phi on the horizontal slice through the electrode
  2. a top-down color map of the same slice with equipotential lines
  3. phi vs distance on log-log axes against the 1/r prediction
Figures are saved to figures/ and opened in a window you can rotate.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fd3d import solve_3d, cell_widths
from grid import make_axis, size_report

mm = 1e-3
I, sigma0 = 50e-6, 1.5            # A, S/m
half = 20 * mm                    # cube is 40 mm on a side
c = half                          # electrode at the center

kw = dict(h_min=50e-6, r_fine=0.5 * mm, growth=1.1, h_max=1 * mm)
x = make_axis(0, 2 * half, c, **kw)
y = make_axis(0, 2 * half, c, **kw)
z = make_axis(0, 2 * half, c, **kw)
size_report(x, y, z)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
R = np.sqrt((X - c)**2 + (Y - c)**2 + (Z - c)**2)
shape = X.shape

i0, j0, l0 = (np.argmin(abs(a - c)) for a in (x, y, z))
source = np.zeros(shape)
dx, dy, dz = cell_widths(x), cell_widths(y), cell_widths(z)
source[i0, j0, l0] = I / (dx[i0] * dy[j0] * dz[l0])      # point current at one node

with np.errstate(divide="ignore"):
    phi_wall = I / (4 * np.pi * sigma0 * R)
phi_wall[i0, j0, l0] = 0.0                                 # never used (interior)

print("\nsolving...")
phi, st = solve_3d(x, y, z, np.full(shape, sigma0), source, phi_wall)
print(f"done: {st['iterations']} CG iterations, "
      f"{st['setup_s'] + st['solve_s']:.1f} s")

# slice through the electrode, zoomed to +/- 5 mm
S = phi[:, :, l0] * 1e3                                    # mV
keep_x = np.abs(x - c) <= 5 * mm
keep_y = np.abs(y - c) <= 5 * mm
xs, ys = (x[keep_x] - c) / mm, (y[keep_y] - c) / mm
S = S[np.ix_(keep_x, keep_y)]
XS, YS = np.meshgrid(xs, ys, indexing="ij")
# 1/r goes to infinity at the electrode, so cut the peak off at r = 0.5 mm
cap = I / (4 * np.pi * sigma0 * 0.5 * mm) * 1e3
Sc = np.minimum(S, cap)

# --- 1. 3D surface ---
fig1 = plt.figure(figsize=(8, 6))
ax = fig1.add_subplot(projection="3d")
ax.plot_surface(XS, YS, Sc, cmap="Blues", linewidth=0, antialiased=True,
                rcount=200, ccount=200)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("potential (mV)")
ax.set_title("Potential on the slice through the electrode\n"
             "(50 µA into 1.5 S/m saline; flat top = cut off at r < 0.5 mm)")
ax.view_init(elev=30, azim=-60)
fig1.tight_layout()
fig1.savefig(ROOT / "figures" / "point_source_surface.png", dpi=200)

# --- 2. top-down map with equipotentials ---
fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
im = ax2.pcolormesh(XS, YS, S, norm=LogNorm(vmin=S.min(), vmax=S.max()),
                    cmap="Blues", shading="auto")
# equipotentials at r = 0.25, 0.5, 1, 2, 4 mm
radii = np.array([0.25, 0.5, 1, 2, 4]) * mm
levels = np.sort(I / (4 * np.pi * sigma0 * radii) * 1e3)
cs = ax2.contour(XS, YS, S, levels=levels, colors="k", linewidths=0.6)
ax2.clabel(cs, fmt="%.1f mV", fontsize=7)
ax2.set_aspect("equal")
ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)")
ax2.set_title("Same slice from above (log color scale)")
fig2.colorbar(im, ax=ax2, label="potential (mV)")
fig2.tight_layout()
fig2.savefig(ROOT / "figures" / "point_source_map.png", dpi=200)

# --- 3. phi vs r against 1/r ---
line = phi[i0 + 1:, j0, l0]
r = x[i0 + 1:] - c
fig3, ax3 = plt.subplots(figsize=(6.5, 4.5))
ax3.loglog(r / mm, I / (4 * np.pi * sigma0 * r) * 1e3, color="0.6", lw=2,
           label="1/r theory")
ax3.loglog(r / mm, line * 1e3, "o", ms=3, color="#1f4e8c", label="solver")
ax3.set_xlabel("distance from electrode (mm)")
ax3.set_ylabel("potential (mV)")
ax3.set_title("Fall-off with distance")
ax3.grid(True, which="both", color="0.9", lw=0.5)
ax3.legend(frameon=False)
fig3.tight_layout()
fig3.savefig(ROOT / "figures" / "point_source_falloff.png", dpi=200)

print("saved 3 figures to figures/")
plt.show()