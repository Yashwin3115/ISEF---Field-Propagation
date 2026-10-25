"""Spherically symmetric solver for (1/r^2) d/dr(r^2 sigma dphi/dr) = -s(r).

Domain r in [r0, R]. Current I enters through the inner sphere r = r0
(a small electrode), and the potential is fixed at the outer sphere r = R.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from fd1d import half_point_sigma


def build_matrix_radial(r, sigma, averaging="harmonic"):
    n = len(r)
    h = r[1] - r[0]
    r_half = 0.5 * (r[:-1] + r[1:])
    w = r_half**2 * half_point_sigma(sigma, averaging)   # r^2 * sigma at half-points

    main = np.zeros(n)
    main[1:-1] = -(w[:-1] + w[1:])
    lower = w.copy()
    upper = w.copy()

    # Row 0: current I flows in at r0 and out through r_{1/2}
    main[0] = -w[0]
    upper[0] = w[0]

    # Row N: fixed potential at the outer boundary
    main[-1] = 1.0
    lower[-1] = 0.0

    return diags([lower, main, upper], [-1, 0, 1], format="csr")


def build_rhs_radial(r, source, current, phi_outer):
    h = r[1] - r[0]
    b = -(r**2) * source * h**2
    b[0] = -current * h / (4 * np.pi)
    b[-1] = phi_outer
    return b


def solve(A, b):
    return spsolve(A, b)