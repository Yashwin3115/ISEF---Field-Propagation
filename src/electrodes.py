"""Electrodes on an insulating substrate (the MEA problem).

Geometry convention
  The substrate is the z = 0 face of the grid. It is never fixed, so it is
  insulating (no current crosses it). Electrodes are patches of nodes on that
  face. The medium (aCSF) fills z > 0.

Stimulation modes -- kept separate on purpose, and recorded in every result:
  "voltage"  stimulating electrode held at V (Dirichlet). The delivered current
             is an OUTPUT, computed by summing the current leaving its nodes.
  "current"  stimulating electrode is a floating metal conductor carrying a
             set current I. Its potential is an OUTPUT.

Return contact
  A patch of nodes held at 0 V: anything given as a boolean mask (a distant
  contact, a ring, another array electrode, outer faces of the domain).

Reference
  Potential is only defined up to a constant; recordings measure differences.
  Every reported potential is phi - phi_ref, with the reference given
  explicitly:
     ("return",)            the return contact (0 V)
     ("point", (x, y, z))   potential interpolated at a point
     ("mask", mask)         mean potential over a patch of nodes

Recording electrodes
  High-impedance, so they draw no current and are not resolved in the grid:
  their potential is interpolated at their coordinates (interpolate_at).
  To test that assumption, a few can be passed as floating conductors with
  zero current instead (recording_conductors).
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fd3d import solve_general, current_out_of, grid_shape


# --------------------------------------------------------------------------
# patches on the substrate (z = 0)
# --------------------------------------------------------------------------

def _on_substrate(x, y, z, inside_xy):
    X, Y = np.meshgrid(x, y, indexing="ij")
    m = np.zeros((len(x), len(y), len(z)), dtype=bool)
    m[:, :, 0] = inside_xy(X, Y)
    return m


def disk(x, y, z, center, radius):
    """Nodes on the substrate within `radius` of `center` (x, y)."""
    cx, cy = center
    tol = 1e-9 * radius
    return _on_substrate(x, y, z, lambda X, Y: (X - cx)**2 + (Y - cy)**2 <= (radius + tol)**2)


def square(x, y, z, center, side):
    """Nodes on the substrate inside an axis-aligned square of the given side."""
    cx, cy = center
    h = side / 2 * (1 + 1e-9)
    return _on_substrate(x, y, z, lambda X, Y: (np.abs(X - cx) <= h) & (np.abs(Y - cy) <= h))


def ring(x, y, z, center, r_inner, r_outer):
    """Nodes on the substrate between two radii (e.g. a surrounding return ring)."""
    cx, cy = center
    def f(X, Y):
        r2 = (X - cx)**2 + (Y - cy)**2
        return (r2 >= r_inner**2) & (r2 <= r_outer**2)
    return _on_substrate(x, y, z, f)


def patch_area(x, y, z, mask):
    """Substrate area (m^2) covered by a patch: sum of its nodes' control-face areas."""
    from fd3d import cell_widths
    dx, dy = cell_widths(x), cell_widths(y)
    A = dx[:, None] * dy[None, :]
    return float(np.sum(A[mask[:, :, 0]]))


# --------------------------------------------------------------------------
# interpolation at recording-electrode coordinates
# --------------------------------------------------------------------------

def interpolate_at(phi, x, y, z, points):
    """Potential at arbitrary (x, y, z) points, shape (n, 3), by trilinear interpolation."""
    f = RegularGridInterpolator((x, y, z), phi, method="linear", bounds_error=True)
    return f(np.atleast_2d(points))


# --------------------------------------------------------------------------
# one stimulation run
# --------------------------------------------------------------------------

def stimulate(x, y, z, sigma, stim, mode, value, return_mask,
              reference=("return",), return_potential=None,
              recording_conductors=(), params=None, rtol=1e-8):
    """Solve one stimulation and return everything as a dict.

    stim          boolean mask of the stimulating electrode
    mode          "voltage" (value = volts) or "current" (value = amps)
    return_mask   boolean mask of nodes held at 0 V (the return contact)
    return_potential   optional full-grid array of potentials for the return
                  nodes instead of 0 (used only by verification tests that
                  apply an exact solution on the outer faces)
    reference     see module docstring
    recording_conductors   masks of passive electrodes to model as floating
                  metal (zero net current) instead of ignoring
    params        dict of assumed parameters, copied into the result
    """
    if mode not in ("voltage", "current"):
        raise ValueError(f'mode must be "voltage" or "current", got {mode!r}')
    if np.any(stim & return_mask):
        raise ValueError("stimulating electrode overlaps the return contact")

    shape = grid_shape(sigma)
    fixed = return_mask.copy()
    phi_fixed = np.zeros(shape) if return_potential is None else return_potential.copy()
    phi_fixed[~return_mask] = 0.0
    floating = [(m, 0.0) for m in recording_conductors]

    if mode == "voltage":
        fixed |= stim
        phi_fixed[stim] = value
    else:
        floating = [(stim, value)] + floating

    phi, st = solve_general(x, y, z, sigma, fixed=fixed, phi_fixed=phi_fixed,
                            floating=floating, rtol=rtol)
    L = st.pop("L")

    I_stim = current_out_of(L, phi, stim)
    I_return = current_out_of(L, phi, return_mask)
    V_stim = float(np.mean(phi[stim]))

    kind = reference[0]
    if kind == "return":
        phi_ref = 0.0
    elif kind == "point":
        phi_ref = float(interpolate_at(phi, x, y, z, np.array(reference[1]))[0])
    elif kind == "mask":
        phi_ref = float(np.mean(phi[reference[1]]))
    else:
        raise ValueError(f"unknown reference {reference!r}")

    return {
        "mode": mode,
        "applied": value,
        "phi": phi,                         # absolute; use phi_rel for anything reported
        "phi_rel": phi - phi_ref,
        "phi_ref": phi_ref,
        "reference": kind if kind == "return" else f"{kind}",
        "stim_potential_V": V_stim,
        "delivered_current_A": I_stim,
        "return_current_A": I_return,       # should be -(delivered current)
        "solver": {k: v for k, v in st.items() if k != "floating_potentials"},
        "params": dict(params or {}),
        "L": L,
    }
