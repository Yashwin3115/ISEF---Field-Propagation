"""3D finite-difference solver for div(sigma grad phi) = -s on a box.

Seven-point stencil. Each grid node carries its own conductivity, so the
node can be thought of as a small cell with one material value (saline,
plastic wall, inner-sphere saline, ...). Conductivity between neighbors uses
the harmonic mean, which test (b) showed is exact at a material jump.

Fixed potential (Dirichlet) on all six faces. Only interior nodes are
unknowns, so the matrix is symmetric positive definite.

Solved with conjugate gradient, preconditioned by algebraic multigrid (pyamg).

Array convention: sigma[i, j, l] is the node at x[i], y[j], z[l].
Unknown numbering: interior node (i, j, l) -> k = ((i-1)*(ny-1) + (j-1))*(nz-1) + (l-1).
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg
import pyamg


def half(sigma, axis, averaging="harmonic"):
    """Conductivity halfway between neighboring nodes along one axis."""
    a = np.take(sigma, np.arange(sigma.shape[axis] - 1), axis=axis)
    c = np.take(sigma, np.arange(1, sigma.shape[axis]), axis=axis)
    return 0.5 * (a + c) if averaging == "arithmetic" else 2 * a * c / (a + c)


def build_system_3d(x, y, z, sigma, source, phi_boundary, averaging="harmonic"):
    """Return A, b for the interior unknowns.
    phi_boundary is a full (nx+1, ny+1, nz+1) array; only its face values are used."""
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    hx, hy, hz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    m = (nx - 1) * (ny - 1) * (nz - 1)

    sx = half(sigma, 0, averaging)   # (nx,   ny+1, nz+1)
    sy = half(sigma, 1, averaging)   # (nx+1, ny,   nz+1)
    sz = half(sigma, 2, averaging)   # (nx+1, ny+1, nz  )

    I, J, L = np.meshgrid(np.arange(1, nx), np.arange(1, ny), np.arange(1, nz),
                          indexing="ij")
    I, J, L = I.ravel(), J.ravel(), L.ravel()

    def k(i, j, l):
        return ((i - 1) * (ny - 1) + (j - 1)) * (nz - 1) + (l - 1)

    K = k(I, J, L)

    neighbors = [
        (sx[I - 1, J, L] / hx**2, I - 1, J, L),   # west
        (sx[I, J, L] / hx**2,     I + 1, J, L),   # east
        (sy[I, J - 1, L] / hy**2, I, J - 1, L),   # south
        (sy[I, J, L] / hy**2,     I, J + 1, L),   # north
        (sz[I, J, L - 1] / hz**2, I, J, L - 1),   # bottom
        (sz[I, J, L] / hz**2,     I, J, L + 1),   # top
    ]

    rows, cols = [K], [K]
    vals = [sum(w for w, *_ in neighbors)]
    b = source[I, J, L].astype(float).copy()

    for w, Ii, Jj, Ll in neighbors:
        on_face = ((Ii == 0) | (Ii == nx) | (Jj == 0) | (Jj == ny) |
                   (Ll == 0) | (Ll == nz))
        b[on_face] += w[on_face] * phi_boundary[Ii[on_face], Jj[on_face], Ll[on_face]]
        inside = ~on_face
        rows.append(K[inside])
        cols.append(k(Ii[inside], Jj[inside], Ll[inside]))
        vals.append(-w[inside])

    A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(m, m)).tocsr()
    return A, b


def solve_cg_amg(A, b, rtol=1e-8):
    """Conjugate gradient with algebraic-multigrid preconditioner.
    Returns the solution and a dict of solver statistics."""
    t0 = time.perf_counter()
    ml = pyamg.smoothed_aggregation_solver(A)
    M = ml.aspreconditioner(cycle="V")
    t_setup = time.perf_counter() - t0

    iters = [0]
    def count(_):
        iters[0] += 1

    t0 = time.perf_counter()
    u, info = cg(A, b, rtol=rtol, M=M, callback=count)
    t_solve = time.perf_counter() - t0

    rel_res = np.linalg.norm(b - A @ u) / np.linalg.norm(b)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}, rel. residual={rel_res:.1e})")
    return u, {"iterations": iters[0], "rel_residual": rel_res,
               "setup_s": t_setup, "solve_s": t_solve}


def solve_3d(x, y, z, sigma, source, phi_boundary, averaging="harmonic", rtol=1e-8):
    """Solve and return (full potential array, solver statistics)."""
    nx, ny, nz = len(x) - 1, len(y) - 1, len(z) - 1
    A, b = build_system_3d(x, y, z, sigma, source, phi_boundary, averaging)
    u, stats = solve_cg_amg(A, b, rtol)
    phi = phi_boundary.astype(float).copy()
    phi[1:nx, 1:ny, 1:nz] = u.reshape(nx - 1, ny - 1, nz - 1)
    stats["unknowns"] = A.shape[0]
    return phi, stats