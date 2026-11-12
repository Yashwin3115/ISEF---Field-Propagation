"""The MaxWell MaxOne array: layout, chip types, and one-pad stimulation runs.

Every number that is not confirmed by Dr. Muller's lab is a setting with a
stated source, and every result records the settings it used.

Confirmed from MaxWell documents
  pitch 17.5 um; 220 columns x 120 rows = 26,400 electrodes; sensing area
  3.85 x 2.10 mm                                     (MaxOne User Guide v2.1)
  MaxOne pad 8.75 x 12.00 um, platinum black; MaxOne+ pad 11.50 x 11.50 um,
  PEDOT                                              (MaxOne User Guide v2.1)
  well inner diameter 19 mm (PSM) or 32 mm (PLM)     (MaxOne User Guide v2.1)
  impedance at 1 kHz: Pt black 300-600 kOhm, PEDOT 70-150 kOhm
                                                     (Electrical Stimulation Guide v1.0)
  voltage pulses, biphasic, positive phase first, 200 us default phase
                                                     (MaxLab Live manual v25.1)

Assumed until the lab answers (Nov 9 email)
  chip: MaxOne (their 400-600 mV matches its recommended 300-600 mV range)
  pad orientation: 8.75 um along x (the 220-column, 3.85 mm direction)
  medium conductivity 1.5 S/m; fluid depth 3.5 mm (about 1 mL in a 19 mm well)
  well modelled as a 19 x 19 mm square box (side walls insulating plastic)
  return path: one of RETURNS below -- run all of them, report the spread

Geometry
  Array centre at x = y = 0. Substrate at z = 0 (insulating). Fluid fills
  0 < z < depth. Only the stimulating pad and the return are resolved in the
  grid; the other electrodes draw no current (high-impedance amplifiers), so
  their potential is interpolated at their centres.
"""
import numpy as np

from grid import graded_axis, size_report
from fd3d import face_mask
from electrodes import interpolate_at

um, mm = 1e-6, 1e-3

PITCH = 17.5 * um
N_COLS, N_ROWS = 220, 120

CHIPS = {
    "MaxOne":  {"pad_x": 8.75 * um, "pad_y": 12.00 * um, "coating": "platinum black",
                "z_1kHz": (300e3, 600e3), "recommended_mV": (300, 600)},
    "MaxOne+": {"pad_x": 11.50 * um, "pad_y": 11.50 * um, "coating": "PEDOT",
                "z_1kHz": (70e3, 150e3), "recommended_mV": (100, 300)},
}

PROVISIONAL = {
    "chip": "MaxOne",
    "sigma_S_per_m": 1.5,
    "depth_m": 3.5 * mm,
    "well_half_width_m": 9.5 * mm,
    "stim_col_row": (110, 60),          # next to the array centre
    "return": "top_plate",
    "anisotropy": None,                 # None = isotropic; or (ax, ay, az) x sigma
    "status": "provisional -- awaiting lab answers (email Nov 9, 2026)",
}

RETURNS = {
    "top_plate":  "fluid surface held at 0 V (a large return above the array); "
                  "side walls insulating",
    "well_wall":  "side walls of the well held at 0 V (a ring return far away); "
                  "fluid surface insulating (air)",
    "periphery":  "a 50 um wide ring on the substrate, 100 um outside the array "
                  "(like the edge reference electrode in Ronchi et al. 2019); "
                  "walls and fluid surface insulating",
}


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def electrode_xy(col, row):
    """Centre (x, y) of electrode (col, row), array centred at the origin."""
    return ((np.asarray(col) - (N_COLS - 1) / 2) * PITCH,
            (np.asarray(row) - (N_ROWS - 1) / 2) * PITCH)


def all_electrodes():
    """(col, row, x, y) for all 26,400 electrodes, as flat arrays."""
    C, R = np.meshgrid(np.arange(N_COLS), np.arange(N_ROWS), indexing="ij")
    X, Y = electrode_xy(C, R)
    return C.ravel(), R.ravel(), X.ravel(), Y.ravel()


def array_extent():
    """Half-widths of the array including the pads (x, y)."""
    return N_COLS * PITCH / 2, N_ROWS * PITCH / 2


# --------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------

RING_GAP, RING_WIDTH = 0.1 * mm, 0.05 * mm     # assumed periphery ring


def build_grid(chip, stim_xy, depth, half_width, n_pad=8, growth=1.10,
               h_array=50 * um, h_far=0.5 * mm, ring=False):
    """Graded grid: about 2 * n_pad cells across the pad's narrow side, growing
    by `growth` per cell, at most h_array over the array (+ 0.2 mm margin),
    at most h_far out to the well walls. ring=True also refines smoothly down
    to RING_WIDTH / 4 * (h_array / 50 um) across the periphery return ring."""
    px, py = CHIPS[chip]["pad_x"] / 2, CHIPS[chip]["pad_y"] / 2
    h = min(px, py) / n_pad
    hx = px / max(1, round(px / h))
    hy = py / max(1, round(py / h))
    ax, ay = array_extent()
    m = 0.2 * mm
    zx = [(-ax - m, ax + m, h_array)]
    zy = [(-ay - m, ay + m, h_array)]
    if ring:
        hr = RING_WIDTH / 4 * (h_array / (50 * um))
        for zs, a in [(zx, ax), (zy, ay)]:
            lo, hi = a + RING_GAP, a + RING_GAP + RING_WIDTH
            zs += [(lo, hi, hr, 1 - 1 / growth), (-hi, -lo, hr, 1 - 1 / growth)]
    x = graded_axis(-half_width, half_width, stim_xy[0], hx, px, growth, h_far, zones=zx)
    y = graded_axis(-half_width, half_width, stim_xy[1], hy, py, growth, h_far, zones=zy)
    hz = min(hx, hy)
    z = graded_axis(0.0, depth, 0.0, hz, round(min(px, py) / hz) * hz, growth, h_far)
    return x, y, z


def rectangle_on_substrate(x, y, z, center, wx, wy):
    tx, ty = wx / 2 * (1 + 1e-9), wy / 2 * (1 + 1e-9)
    X, Y = np.meshgrid(x, y, indexing="ij")
    m = np.zeros((len(x), len(y), len(z)), dtype=bool)
    m[:, :, 0] = (np.abs(X - center[0]) <= tx) & (np.abs(Y - center[1]) <= ty)
    return m


def return_mask(kind, x, y, z):
    shape = (len(x), len(y), len(z))
    if kind == "top_plate":
        return face_mask(shape, ["z+"])
    if kind == "well_wall":
        return face_mask(shape, ["x-", "x+", "y-", "y+"])
    if kind == "periphery":
        # A node belongs to the ring if its cell (control area) touches the
        # ring. Selecting by node position alone can leave a side of a thin ring
        # with no nodes at all on a coarse grid, i.e. a ring with gaps --
        # a different return path on every grid (found Nov 9, 2026).
        ax, ay = array_extent()
        gap, w = RING_GAP, RING_WIDTH
        def cell_lo_hi(a):
            lo, hi = a.copy(), a.copy()
            lo[1:] = 0.5 * (a[1:] + a[:-1])
            hi[:-1] = 0.5 * (a[1:] + a[:-1])
            return lo, hi
        xl, xh = cell_lo_hi(x)
        yl, yh = cell_lo_hi(y)
        XL, YL = np.meshgrid(xl, yl, indexing="ij")
        XH, YH = np.meshgrid(xh, yh, indexing="ij")
        ox, oy = ax + gap + w, ay + gap + w           # outer edge of the ring
        ix, iy = ax + gap, ay + gap                   # inner edge of the ring
        touches_outer = (XH > -ox) & (XL < ox) & (YH > -oy) & (YL < oy)
        inside_inner = (XL >= -ix) & (XH <= ix) & (YL >= -iy) & (YH <= iy)
        m = np.zeros(shape, dtype=bool)
        m[:, :, 0] = touches_outer & ~inside_inner
        return m
    raise ValueError(f"unknown return {kind!r}; choose from {list(RETURNS)}")


# --------------------------------------------------------------------------
# one stimulation run: transfer field per unit current
# --------------------------------------------------------------------------

def transfer_run(params=None, n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm,
                 I_test=1e-6, verbose=True):
    """Drive the stimulating pad with I_test (current mode, floating metal pad)
    and return the potential at every electrode per ampere.

    Because the equation is linear, every voltage-mode or time-varying result
    is this field times the actual current (see interface.py).
    """
    from electrodes import stimulate
    p = {**PROVISIONAL, **(params or {})}
    chip = CHIPS[p["chip"]]
    sx, sy = electrode_xy(*p["stim_col_row"])
    x, y, z = build_grid(p["chip"], (sx, sy), p["depth_m"], p["well_half_width_m"],
                         n_pad, growth, h_array, h_far, ring=(p["return"] == "periphery"))
    shape = (len(x), len(y), len(z))
    stim = rectangle_on_substrate(x, y, z, (sx, sy), chip["pad_x"], chip["pad_y"])
    ret = return_mask(p["return"], x, y, z)
    if verbose:
        print(f"chip {p['chip']} ({chip['pad_x']/um:.2f} x {chip['pad_y']/um:.2f} um), "
              f"stim pad {p['stim_col_row']}, return '{p['return']}'")
        size_report(x, y, z, n_unknowns=int(np.prod(shape) - ret.sum() - stim.sum() + 1))

    a = p.get("anisotropy")            # (ax, ay, az) multipliers of sigma, or None
    if a is None:
        sigma = np.full(shape, p["sigma_S_per_m"])
    else:
        sigma = tuple(np.full(shape, p["sigma_S_per_m"] * float(f)) for f in a)
    r = stimulate(x, y, z, sigma, stim, "current", I_test, ret,
                  reference=("return",), params=p)

    col, row, ex, ey = all_electrodes()
    pts = np.column_stack([ex, ey, np.zeros_like(ex)])
    Z = interpolate_at(r["phi_rel"], x, y, z, pts) / I_test       # ohms
    dist = np.hypot(ex - sx, ey - sy)
    is_stim = (col == p["stim_col_row"][0]) & (row == p["stim_col_row"][1])
    Z[is_stim] = r["stim_potential_V"] / I_test   # the pad itself: its own potential

    return {
        "params": p,
        "grid": {"n_pad": n_pad, "growth": growth, "h_array_m": h_array, "h_far_m": h_far,
                 "nodes": [len(x), len(y), len(z)], "unknowns": r["solver"]["unknowns"]},
        "solver": r["solver"],
        "R_solution_ohm": r["stim_potential_V"] / I_test,     # pad to return, through the fluid
        "current_check": r["delivered_current_A"] / I_test,   # should be 1
        "return_check": -r["return_current_A"] / I_test,     # should be 1
        "col": col, "row": row, "x": ex, "y": ey, "distance_m": dist,
        "transfer_ohm": Z,                                     # phi_rel per ampere
        "x_grid": x, "y_grid": y, "z_grid": z, "phi_per_A": r["phi_rel"] / I_test,
    }
