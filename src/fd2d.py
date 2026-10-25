"""2D finite-difference solver for div(sigma grad phi) = -s on a rectangle.

Five-point stencil, conductivity given at every node, harmonic averaging at
the half-points, fixed potential (Dirichlet) on all four edges.

Only interior nodes are unknowns. Boundary values are moved to the right-hand
side, which keeps the matrix symmetric and positive definite -- the property
conjugate gradient needs later in 3D.

Array convention: sigma[i, j] is the node at x[i], y[j].
Unknown numbering: interior node (i, j) is unknown k = (i-1)*(ny-1) + (j-1).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def half_x(sigma, averaging="harmonic"):
    """Conductivity between x[i] and x[i+1]. Shape (nx, ny+1)."""
    a, c = sigma[:-1, :], sigma[1:, :]
    return 0.5 * (a + c) if averaging == "arithmetic" else 2 * a * c / (a + c)


def half_y(sigma, averaging="harmonic"):
    """Conductivity between y[j] and y[j+1]. Shape (nx+1, ny)."""
    a, c = sigma[:, :-1], sigma[:, 1:]
    return 0.5 * (a + c) if averaging == "arithmetic" else 2 * a * c / (a + c)


def build_system_2d(x, y, sigma, source, phi_boundary, averaging="harmonic"):
    """Return A, b for the interior unknowns.
    phi_boundary is a full (nx+1, ny+1) array; only its edge values are used."""
    nx, ny = len(x) - 1, len(y) - 1
    hx, hy = x[1] - x[0], y[1] - y[0]
    m = (nx - 1) * (ny - 1)

    sx = half_x(sigma, averaging)
    sy = half_y(sigma, averaging)

    # every interior node, flattened in the same order as the unknowns
    I, J = np.meshgrid(np.arange(1, nx), np.arange(1, ny), indexing="ij")
    I, J = I.ravel(), J.ravel()

    def k(i, j):
        return (i - 1) * (ny - 1) + (j - 1)

    K = k(I, J)

    # stencil weights to the west, east, south, north neighbors
    wW = sx[I - 1, J] / hx**2
    wE = sx[I, J] / hx**2
    wS = sy[I, J - 1] / hy**2
    wN = sy[I, J] / hy**2

    rows, cols, vals = [K], [K], [wW + wE + wS + wN]
    b = source[I, J].astype(float).copy()

    for w, Ii, Jj in [(wW, I - 1, J), (wE, I + 1, J), (wS, I, J - 1), (wN, I, J + 1)]:
        on_edge = (Ii == 0) | (Ii == nx) | (Jj == 0) | (Jj == ny)
        # neighbor is a boundary node: its known value moves to the right-hand side
        b[on_edge] += w[on_edge] * phi_boundary[Ii[on_edge], Jj[on_edge]]
        # neighbor is an unknown: off-diagonal entry
        inside = ~on_edge
        rows.append(K[inside])
        cols.append(k(Ii[inside], Jj[inside]))
        vals.append(-w[inside])

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(m, m)).tocsr()
    return A, b


def solve_2d(x, y, sigma, source, phi_boundary, averaging="harmonic"):
    """Solve and return the full (nx+1, ny+1) potential, boundary included."""
    nx, ny = len(x) - 1, len(y) - 1
    A, b = build_system_2d(x, y, sigma, source, phi_boundary, averaging)
    phi = phi_boundary.astype(float).copy()
    phi[1:nx, 1:ny] = spsolve(A, b).reshape(nx - 1, ny - 1)
    return phi