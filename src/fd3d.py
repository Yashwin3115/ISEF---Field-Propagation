"""3D finite-volume solver for div(sigma grad phi) = -s.

Grid
  Rectangular, nonuniform: spacing may change from cell to cell along each
  axis. Each node owns a small box (its control volume) reaching halfway to
  each neighbor; nodes on the outer faces own half (or quarter, or eighth)
  boxes. Current between two neighboring nodes is

      G * (potential difference),   G = sigma_face * face_area / distance.

  The conductance from A to B is the same number as from B to A, so every
  matrix here is symmetric. Conductivity at a face uses the harmonic mean,
  which test (b) showed is exact at a material jump.

The conductance matrix L
  L is built once for ALL nodes. (L @ phi)[i] is the net current flowing OUT
  of node i into its neighbors. Everything else follows from it:
    - free node:      (L phi)_i = injected current (0 in plain saline)
    - fixed node:     (L phi)_i = current that the electrode/contact delivers there
    - sum over a set: net current leaving that set

Three kinds of node
  free      ordinary medium; potential unknown.
  fixed     Dirichlet: potential is set. Used for a stimulating electrode in
            VOLTAGE mode, for the return contact, and for outer faces carrying
            a known potential.
  floating  a metal conductor that is not wired to a voltage: all its nodes
            share ONE unknown potential, and a set net current flows through
            it. Used for a stimulating electrode in CURRENT mode (current = I)
            and for passive recording electrodes (current = 0).

Insulating surfaces (Neumann, zero current)
  Any outer face that is not fixed is insulating automatically, because there
  is no neighbor on the other side for current to reach. The MEA substrate at
  z = 0 and symmetry planes are handled this way.

Solved with conjugate gradient + classical algebraic multigrid (pyamg).
Array convention: sigma[i, j, l] is the node at x[i], y[j], z[l]; flat index
is C order of the full grid.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import pyamg


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------

def half(sigma, axis, averaging="harmonic"):
    """Conductivity halfway between neighboring nodes along one axis."""
    a = np.take(sigma, np.arange(sigma.shape[axis] - 1), axis=axis)
    c = np.take(sigma, np.arange(1, sigma.shape[axis]), axis=axis)
    return 0.5 * (a + c) if averaging == "arithmetic" else 2 * a * c / (a + c)


def cell_widths(x):
    """Width of each node's control volume: half the gap on each side."""
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def node_volumes(x, y, z):
    dx, dy, dz = cell_widths(x), cell_widths(y), cell_widths(z)
    return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]


def face_mask(shape, faces="all"):
    """Boolean mask of nodes on the chosen outer faces.
    faces: "all", or a list from {"x-", "x+", "y-", "y+", "z-", "z+"}."""
    if faces == "all":
        faces = ["x-", "x+", "y-", "y+", "z-", "z+"]
    m = np.zeros(shape, dtype=bool)
    for f in faces:
        axis = "xyz".index(f[0])
        idx = [slice(None)] * 3
        idx[axis] = 0 if f[1] == "-" else -1
        m[tuple(idx)] = True
    return m


# --------------------------------------------------------------------------
# the conductance matrix
# --------------------------------------------------------------------------

def conductance_matrix(x, y, z, sigma, averaging="harmonic"):
    """Symmetric N x N matrix L over ALL nodes: (L phi)_i = current out of node i."""
    shape = sigma.shape
    N = sigma.size
    idx = np.arange(N).reshape(shape)
    dx, dy, dz = cell_widths(x), cell_widths(y), cell_widths(z)
    gaps = (np.diff(x), np.diff(y), np.diff(z))
    areas = (dy[None, :, None] * dz[None, None, :],
             dx[:, None, None] * dz[None, None, :],
             dx[:, None, None] * dy[None, :, None])

    rows, cols, vals = [], [], []
    diag = np.zeros(N)
    for axis in range(3):
        n = shape[axis]
        p_sl = [slice(None)] * 3; p_sl[axis] = slice(0, n - 1)
        q_sl = [slice(None)] * 3; q_sl[axis] = slice(1, n)
        p_sl, q_sl = tuple(p_sl), tuple(q_sl)
        gshape = [1, 1, 1]; gshape[axis] = n - 1
        G = half(sigma, axis, averaging) * areas[axis][p_sl] / gaps[axis].reshape(gshape)
        G = np.broadcast_to(G, idx[p_sl].shape).ravel()
        p, q = idx[p_sl].ravel(), idx[q_sl].ravel()
        rows += [p, q]
        cols += [q, p]
        vals += [-G, -G]
        diag += np.bincount(p, G, N) + np.bincount(q, G, N)

    rows.append(np.arange(N)); cols.append(np.arange(N)); vals.append(diag)
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N)).tocsr()


# --------------------------------------------------------------------------
# linear solver
# --------------------------------------------------------------------------

def solve_cg_amg(A, b, rtol=1e-8, maxiter=2000):
    """Conjugate gradient with algebraic-multigrid preconditioner.
    Returns the solution and a dict of solver statistics."""
    if not np.any(b):                       # nothing drives any current
        return np.zeros_like(b), {"iterations": 0, "rel_residual": 0.0,
                                  "setup_s": 0.0, "solve_s": 0.0}
    t0 = time.perf_counter()
    # Classical (Ruge-Stuben) multigrid. On the MEA grids, where cells near the
    # pad are 0.5 um wide in one direction and up to 0.5 mm in another, it
    # needed 35 CG iterations vs 266 for smoothed aggregation (7x faster,
    # Nov 9, 2026 comparison on a 1.1M-unknown array problem).
    ml = pyamg.ruge_stuben_solver(A)
    M = ml.aspreconditioner(cycle="V")
    t_setup = time.perf_counter() - t0

    iters = [0]
    def count(_):
        iters[0] += 1

    t0 = time.perf_counter()
    u, info = cg(A, b, rtol=rtol, M=M, callback=count, maxiter=maxiter)
    t_solve = time.perf_counter() - t0

    rel_res = np.linalg.norm(b - A @ u) / np.linalg.norm(b)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info}, rel. residual={rel_res:.1e})")
    return u, {"iterations": iters[0], "rel_residual": rel_res,
               "setup_s": t_setup, "solve_s": t_solve}


# --------------------------------------------------------------------------
# input checks
# --------------------------------------------------------------------------

def check_inputs(x, y, z, sigma, source, phi_fixed, fixed, floating=()):
    """Stop with a clear message if the inputs can't give a meaningful answer."""
    shape = (len(x), len(y), len(z))
    for name, a in [("x", x), ("y", y), ("z", z)]:
        if len(a) < 3:
            raise ValueError(f"{name} needs at least 3 nodes, got {len(a)}")
        if not np.all(np.diff(a) > 0):
            raise ValueError(f"{name} must be strictly increasing (no repeated or "
                             f"backwards nodes)")
    for name, a in [("sigma", sigma), ("source", source), ("phi_fixed", phi_fixed),
                    ("fixed", fixed)]:
        if a.shape != shape:
            raise ValueError(f"{name} has shape {a.shape}, grid is {shape}")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma must be positive and finite everywhere "
                         f"(min found: {np.nanmin(sigma):.3g})")
    if not np.all(np.isfinite(source)):
        raise ValueError("source contains NaN or infinity")
    if not fixed.any():
        raise ValueError("at least one fixed-potential node is needed "
                         "(a return contact or a fixed outer face)")
    if not np.all(np.isfinite(phi_fixed[fixed])):
        raise ValueError("phi_fixed has NaN or infinity at a fixed node")
    taken = fixed.copy()
    for k, (mask, current) in enumerate(floating):
        if mask.shape != shape:
            raise ValueError(f"floating conductor {k} has shape {mask.shape}, grid is {shape}")
        if not mask.any():
            raise ValueError(f"floating conductor {k} contains no nodes")
        if np.any(mask & taken):
            raise ValueError(f"floating conductor {k} overlaps a fixed node or another conductor")
        if not np.isfinite(current):
            raise ValueError(f"floating conductor {k} has a non-finite current")
        taken |= mask


# --------------------------------------------------------------------------
# general solve
# --------------------------------------------------------------------------

def solve_general(x, y, z, sigma, source=None, fixed=None, phi_fixed=None,
                  floating=(), averaging="harmonic", rtol=1e-8):
    """Solve with any mix of fixed, floating and free nodes.

    sigma      conductivity (S/m) at every node
    source     current density (A/m^3) at every node; the current injected at a
               node is source * (control volume). Default: none.
    fixed      boolean mask of fixed-potential nodes. Default: all six outer faces.
    phi_fixed  potential at fixed nodes (only those entries are used). Default 0.
    floating   list of (mask, net_current_A): each is one equipotential metal
               conductor with that net current flowing out of it into the medium.

    Returns (phi, info). info["L"] is the conductance matrix, kept so delivered
    currents can be computed afterwards (see current_out_of).
    """
    shape = sigma.shape
    if source is None:
        source = np.zeros(shape)
    if fixed is None:
        fixed = face_mask(shape)
    if phi_fixed is None:
        phi_fixed = np.zeros(shape)
    floating = list(floating)
    check_inputs(x, y, z, sigma, source, phi_fixed, fixed, floating)

    L = conductance_matrix(x, y, z, sigma, averaging)
    N = sigma.size
    fixed_f = fixed.ravel()
    phiF = np.where(fixed_f, phi_fixed.ravel(), 0.0)
    inj = (source * node_volumes(x, y, z)).ravel()

    # column of the reduced system that each node maps to (-1 = fixed)
    col = np.full(N, -1, dtype=np.int64)
    free = ~fixed_f
    for mask, _ in floating:
        free &= ~mask.ravel()
    n_free = int(free.sum())
    col[free] = np.arange(n_free)
    for k, (mask, _) in enumerate(floating):
        col[mask.ravel()] = n_free + k
    m = n_free + len(floating)

    nf = np.flatnonzero(~fixed_f)                     # every non-fixed node
    P = csr_matrix((np.ones(nf.size), (np.arange(nf.size), col[nf])), shape=(nf.size, m))
    L_nn = L[nf][:, nf]
    A = (P.T @ L_nn @ P).tocsr()
    rhs = inj[nf] - L[nf] @ phiF                      # move known potentials across
    b = P.T @ rhs
    for k, (_, current) in enumerate(floating):
        b[n_free + k] += current

    u, stats = solve_cg_amg(A, b, rtol)
    if not np.all(np.isfinite(u)):
        raise RuntimeError("solution contains NaN or infinity")

    phi = phiF.copy()
    phi[nf] = P @ u
    phi = phi.reshape(shape)
    stats.update(unknowns=m, L=L,
                 floating_potentials=[float(u[n_free + k]) for k in range(len(floating))])
    return phi, stats


def current_out_of(L, phi, mask):
    """Net current (A) flowing out of the nodes in `mask` into the rest of the medium.
    For a stimulating electrode this is the delivered current.
    For a region of free nodes enclosing the electrode it is the flux through
    the region's surface."""
    return float(np.sum((L @ phi.ravel())[mask.ravel()]))


# --------------------------------------------------------------------------
# backward-compatible interface (used by the earlier tests and step 7)
# --------------------------------------------------------------------------

def build_system_3d(x, y, z, sigma, source, phi_fixed, fixed=None, averaging="harmonic"):
    """Return A, b, unknown_mask for the free nodes (no floating conductors)."""
    if fixed is None:
        fixed = face_mask(sigma.shape)
    if not fixed.any():
        raise ValueError("at least one fixed-potential node is needed")
    L = conductance_matrix(x, y, z, sigma, averaging)
    unknown = ~fixed
    u = np.flatnonzero(unknown.ravel())
    phiF = np.where(fixed, phi_fixed, 0.0).ravel()
    inj = (source * node_volumes(x, y, z)).ravel()
    A = L[u][:, u].tocsr()
    b = inj[u] - L[u] @ phiF
    return A, b, unknown


def solve_3d(x, y, z, sigma, source, phi_fixed, fixed=None, averaging="harmonic", rtol=1e-8):
    """Solve with fixed nodes only; return (full potential array, solver statistics)."""
    if fixed is None:
        fixed = face_mask(sigma.shape)
    phi, st = solve_general(x, y, z, sigma, source, fixed, phi_fixed,
                            averaging=averaging, rtol=rtol)
    st.pop("L")
    st.pop("floating_potentials")
    return phi, st
