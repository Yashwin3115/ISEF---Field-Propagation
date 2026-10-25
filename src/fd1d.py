"""1D finite-difference solver for d/dx(sigma dphi/dx) = -s(x)."""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def half_point_sigma(sigma, averaging="harmonic"):
    """Conductivity at the midpoints between neighboring nodes."""
    a, b = sigma[:-1], sigma[1:]
    if averaging == "arithmetic":
        return 0.5 * (a + b)
    if averaging == "harmonic":
        return 2 * a * b / (a + b)
    raise ValueError(f"unknown averaging: {averaging}")


def build_matrix(sigma, averaging="harmonic"):
    """Tridiagonal matrix with Dirichlet rows at both ends."""
    n = len(sigma)
    sh = half_point_sigma(sigma, averaging)   # length n-1

    main = np.ones(n)
    main[1:-1] = -(sh[:-1] + sh[1:])

    lower = sh.copy()      # row i gets sigma_{i-1/2}
    lower[-1] = 0.0        # last row is a boundary row
    upper = sh.copy()      # row i gets sigma_{i+1/2}
    upper[0] = 0.0         # first row is a boundary row

    return diags([lower, main, upper], [-1, 0, 1], format="csr")


def build_rhs(source, h, phi_left, phi_right):
    b = -source * h**2
    b[0] = phi_left
    b[-1] = phi_right
    return b


def solve(A, b):
    return spsolve(A, b)