"""Shared setup for verification tests V1-V4 and V7: a disk electrode flush with
an insulating plane, in a half space of uniform saline.

Exact solution (equipotential disk of radius a at potential V0, Newman 1966):
    phi(r, z) = (2 V0 / pi) * arcsin( 2a / ( sqrt((r-a)^2 + z^2) + sqrt((r+a)^2 + z^2) ) )
On the axis this is (2 V0 / pi) arctan(a / z). The delivered current is
I = 4 sigma a V0 (spreading resistance R = 1/(4 sigma a)), so equivalently
    phi(0, z) = (I / (2 pi sigma a)) arctan(a / z).

Symmetry: the disk is centred at the origin, so only one quarter (x >= 0,
y >= 0) is solved. The x = 0 and y = 0 faces are symmetry planes and the z = 0
face is the insulating substrate -- all three are left insulating. The quarter
carries 1/4 of the current. The other three faces either carry the exact
potential (so cutting the half space down to a box adds no error) or are
grounded at 0 V (a simple return contact).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from grid import make_axis
from fd3d import face_mask
from electrodes import disk, stimulate

um = 1e-6
A_RADIUS = 6.5 * um        # provisional MaxOne equivalent electrode radius
SIGMA = 1.5                # S/m, provisional aCSF
QUARTER = 4                # full-disk quantities = 4 x quarter quantities


def exact_phi(R, Z, V0, a=A_RADIUS):
    s = np.sqrt((R - a)**2 + Z**2) + np.sqrt((R + a)**2 + Z**2)
    return 2 * V0 / np.pi * np.arcsin(np.minimum(2 * a / s, 1.0))


def grid(n=20, size=30, a=A_RADIUS):
    """Quarter-domain grid: spacing a/n over the disk, growing 10% per cell,
    box edge `size` * a."""
    L = size * a
    ax = make_axis(0, L, 0, h_min=a / n, r_fine=1.3 * a, growth=1.1, h_max=L / 8)
    return ax, ax.copy(), ax.copy()


def solve_disk(mode, value, n=20, size=30, outer="exact", a=A_RADIUS, sigma0=SIGMA):
    """mode "voltage" (value = V0) or "current" (value = full-disk current I).
    outer = "exact": outer faces carry the exact half-space potential.
    outer = "ground": outer faces held at 0 V."""
    x, y, z = grid(n, size, a)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    sigma = np.full(X.shape, sigma0)
    stim = disk(x, y, z, (0.0, 0.0), a)
    ret = face_mask(X.shape, ["x+", "y+", "z+"])

    if outer == "exact":
        V0 = value if mode == "voltage" else value / (4 * sigma0 * a)
        ret_phi = exact_phi(np.hypot(X, Y), Z, V0, a)
    else:
        ret_phi = None
    applied = value if mode == "voltage" else value / QUARTER

    r = stimulate(x, y, z, sigma, stim, mode, applied, ret,
                  return_potential=ret_phi,
                  params={"sigma_S_per_m": sigma0, "radius_m": a, "n": n,
                          "outer": outer, "symmetry": "quarter"})
    r["delivered_current_A"] *= QUARTER
    r["return_current_A"] *= QUARTER
    r.update(x=x, y=y, z=z, stim=stim, ret=ret)
    return r
