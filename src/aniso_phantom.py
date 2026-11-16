"""Anisotropic phantom (experiment 8): a printed block of parallel insulating
plates with saline gaps, and slots through the plates.

  Along the plates the current flows freely in the gaps; across the plates it
  can only pass through the slots. The block therefore conducts better along
  the plates than across them: an ANISOTROPIC medium built from an isotropic
  liquid, with a ratio set by the slot width and pitch.

Axes (block centred on the stimulating disk, standing on the insulating plate)
  x  along the plates (the easy direction)
  y  across the plates (the hard direction)
  z  up; plates, gaps and slots all run the full height of the block

Three predictions of the same experiment
  isotropic   no block (the uniform tank)
  tensor      the block replaced by a uniform medium with the effective
              conductivities (sigma_x, sigma_y, sigma_z) from the unit cell
  resolved    every plate, gap and slot in the grid

Exact geometry on the grid
  The solver's material boundaries lie midway between nodes. The block grid
  uses a spacing h = 1 mm / (odd number), with nodes at whole multiples of h,
  so every boundary at a half-millimetre (plate faces at +-0.5, +-1.5, ...,
  slot edges at centre +- 0.5 mm) falls exactly midway between two nodes at
  EVERY refinement level. Refining by odd factors (1, 1/3, 1/5, 1/7 mm) keeps
  the geometry identical and only resolves the field better. This is why the
  slot width is an odd whole number of mm and the slot pitch an even one.

Slots are STAGGERED: the slots of neighbouring plates are offset by half a
pitch, so the current must run sideways through a gap to reach the next slot.
(Aligned slots form straight open channels across the whole block; reaching
9:1 that way needs so few slots that the block is no longer a medium.)
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fd3d import solve_general, current_out_of
from grid import graded_axis, one_side
from phantom import (tank_geometry, tank_masks, layout_setup, mm)

ANISO_DEFAULT = {
    "_comment": "Anisotropic block (experiment 8, built last). ASSUMED until printed and measured. "
                "Keep: plates and gaps 1.0 mm, slot width an odd whole number of mm (1 or 3), slot pitch "
                "an even whole number of mm. A variant may set its own slot_width_mm.",
    "plate_thickness_mm": 1.0, "gap_mm": 1.0, "slot_width_mm": 1.0,
    "block_x_mm": 39.0,                      # along the plates (half-length must end on a half-millimetre)
    "n_plates_each_side": 10,                # plates at y = +-1, +-3, ... (the disk sits in the central gap)
    "block_height_mm": 40.0,
    "staggered": True,
    # chosen by runs/aniso_design.py: 3 mm slots every 8 mm -> 2.9:1; 1 mm slots every 10 mm -> 9.2:1
    "variants": {"3to1": {"slot_pitch_mm": 8.0, "slot_width_mm": 3.0}, "9to1": {"slot_pitch_mm": 10.0}},
    "along_distances_mm": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 18.0, 25.0, 30.0],   # in the central gap, +x
    "across_distances_mm": [2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0, 25.0, 30.0],            # gap centres, +y
    "recording_tip_z_mm": [0.5, 1.5],
}


# ---------------------------------------------------------------------------
# unit cell: effective conductivity across the plates
# ---------------------------------------------------------------------------

def unit_cell_sigma_y(pitch_mm, slot_mm=1.0, plate_mm=1.0, gap_mm=1.0, staggered=True, n=5):
    """sigma_y / sigma of the periodic structure from one cell, solved with the
    project's own solver (h = 1 mm / n, n odd).

    Cell: x from a slot centre (0) to half a pitch; y from the mid-plane of one
    plate to the mid-plane of the next. Both mid-planes are equipotentials by
    mirror symmetry (inside the slots; elsewhere they are plastic), the sides
    are symmetry planes (no current). Staggered: the second plate's slot is at
    x = pitch / 2; aligned: at x = 0."""
    if n % 2 == 0:
        raise ValueError("n must be odd (boundaries at half-millimetres must fall between nodes)")
    h = 1.0 / n
    x = np.arange(0, pitch_mm / 2 + 1e-9, h)
    y = np.arange(0, plate_mm + gap_mm + 1e-9, h)
    z = np.array([0.0, h, 2 * h])
    X, Y = np.meshgrid(x, y, indexing="ij")
    in_gap = (Y > plate_mm / 2) & (Y < plate_mm / 2 + gap_mm)
    slotA = X < slot_mm / 2
    slotB = (X > pitch_mm / 2 - slot_mm / 2) if staggered else slotA
    open_ = in_gap | ((Y < plate_mm / 2) & slotA) | ((Y > plate_mm / 2 + gap_mm) & slotB)
    sig = np.where(open_, 1.0, 1e-9)[:, :, None] * np.ones(3)[None, None, :]
    fixed = np.zeros(sig.shape, bool); pf = np.zeros(sig.shape)
    fixed[:, 0, :] = (slotA & (Y == 0))[:, 0][:, None]
    fixed[:, -1, :] = (slotB & (Y == Y.max()))[:, -1][:, None]
    pf[:, 0, :] = 1.0
    phi, st = solve_general(x * mm, y * mm, z * mm, sig, fixed=fixed, phi_fixed=pf, rtol=1e-10)
    top = np.zeros(sig.shape, bool); top[:, 0, :] = fixed[:, 0, :]
    I = current_out_of(st["L"], phi, top)
    area = (pitch_mm / 2 * mm) * (2 * h * mm)
    return I * ((plate_mm + gap_mm) * mm) / area


def effective_tensor(pitch_mm, slot_mm=1.0, plate_mm=1.0, gap_mm=1.0, staggered=True, n=7):
    """(sigma_x, sigma_y, sigma_z) / sigma of the block as a uniform medium.
    x: the gaps conduct, the plates do not (slots are side pockets): g / (g + p).
    z: gaps plus the slots' open area. y: from the unit cell."""
    s = slot_mm / pitch_mm
    return (gap_mm / (gap_mm + plate_mm),
            float(unit_cell_sigma_y(pitch_mm, slot_mm, plate_mm, gap_mm, staggered, n)),
            (gap_mm + plate_mm * s) / (gap_mm + plate_mm))


# ---------------------------------------------------------------------------
# the block in the tank
# ---------------------------------------------------------------------------

def block_axis(lo, hi, c, half_block, h, core, growth, h_far, one_sided=False):
    """Nodes on [lo, hi] with a node at c. Within |s - c| <= half_block + 1 mm the
    nodes are at whole multiples of h (h = 1 mm / odd), refined by odd factors
    near the electrode: h/9 within core[0], h/3 within core[1] -- odd factors
    keep every half-millimetre boundary midway between two nodes, PROVIDED the
    two core radii are whole millimetres (so each zone ends on a node of the
    next, coarser lattice). solve_block checks this on every grid. Outside the
    block the spacing grows by `growth` per cell up to h_far."""
    def side(D):
        pts = [0.0]
        edge = half_block + 1 * mm
        for r_out, hh in ((core[0], h / 9), (core[1], h / 3), (edge, h)):
            n = int(round((r_out - pts[-1]) / hh))
            pts += [pts[-1] + hh * (k + 1) for k in range(n)]
        if D > pts[-1] + 1e-12:
            out = one_side(D - pts[-1], h, 0.0, growth, h_far)
            pts += list(pts[-1] + out[1:])
        return np.array(pts)
    right = c + side(hi - c)
    if one_sided:
        return right
    left = c - side(c - lo)[::-1]
    return np.concatenate([left, right[1:]])


def faces_offset(s_mm, half_mm):
    """Largest distance (mm) from any half-millimetre boundary within
    |s| <= half_mm to the nearest midpoint between two neighbouring nodes.
    Zero means every plate face and slot edge is represented exactly."""
    mid = (s_mm[1:] + s_mm[:-1]) / 2
    faces = np.arange(0.5, half_mm + 1e-9, 1.0)
    faces = np.concatenate([-faces[::-1], faces]) if s_mm.min() < -0.25 else faces
    return float(max(np.min(np.abs(mid - f)) for f in faces))


def aniso_geometry(setup, variant):
    an = setup["anisotropic"]
    v = dict(an); v.update(an["variants"][variant])
    s6 = layout_setup(setup, None)
    geo = tank_geometry(s6)
    if not geo["sym"]:
        raise ValueError("the anisotropic block run needs the electrode and the return on the tank's "
                         "mid-plane (y = Ly / 2)")
    hx = v["block_x_mm"] / 2
    if abs((hx - 0.5) - round(hx - 0.5)) > 1e-9:
        raise ValueError("block_x_mm / 2 must end on a half-millimetre (e.g. 39 -> 19.5)")
    if abs(v["plate_thickness_mm"] - 1) > 1e-9 or abs(v["gap_mm"] - 1) > 1e-9:
        raise ValueError("this grid represents 1.0 mm plates and gaps exactly; other sizes need "
                         "a different grid (ask before changing them)")
    w = v["slot_width_mm"]
    if abs(w - round(w)) > 1e-9 or int(round(w)) % 2 == 0:
        raise ValueError("slot_width_mm must be an odd whole number of mm (1 or 3), so that the slot "
                         "edges fall on half-millimetres")
    P = v["slot_pitch_mm"]
    if abs(P / 2 - round(P / 2)) > 1e-9:
        raise ValueError("slot_pitch_mm must be an even whole number of mm")
    if v.get("staggered", True) and w >= P / 2:
        raise ValueError("staggered slots must be narrower than half the pitch (the disk must face plastic)")
    return v, s6, geo


def block_material(v, X_mm, Y_mm):
    """True where (x, y) relative to the disk is PLASTIC. Plates are centred at
    |y| = 1, 3, 5, ... mm. Slots: plates 1, 3, 5, ... (counting out from the
    disk) have slots centred at x = P/2 + k P; plates 2, 4, ... at x = k P
    (staggered), so the disk faces solid plastic on both sides. Aligned: all
    at x = P/2 + k P."""
    P = v["slot_pitch_mm"]
    ay, ax = np.abs(Y_mm), np.abs(X_mm)
    j = np.floor((ay + 1) / 2).astype(int)               # plate index 1.. for |y| in (0.5, 1.5), (2.5, 3.5), ...
    in_plate = (np.abs(ay - (2 * j - 1)) < 0.5) & (j >= 1) & (j <= v["n_plates_each_side"])
    in_plate &= ax < v["block_x_mm"] / 2
    odd = (j % 2 == 1) | (not v.get("staggered", True))
    centre_shift = np.where(odd, P / 2, 0.0)
    d = np.abs(((X_mm - centre_shift + P / 2) % P) - P / 2)   # distance to the nearest slot centre
    slot = d < v["slot_width_mm"] / 2
    return in_plate & ~slot


def solve_block(setup, variant, n=5, model="resolved", tensor=None, verbose=True, pos_sd_mm=0.1):
    """One anisotropic-phantom solve. n: block grid h = 1 mm / n (odd).
    model: 'resolved' (every plate and slot), 'tensor' (uniform block with
    conductivities `tensor` x sigma), 'isotropic' (no block).
    Returns G (tip-averaged, condition A and the far-reference value) along
    the plates (+x, central gap) and across them (+y, gap centres)."""
    if n % 2 == 0:
        raise ValueError("n must be odd")
    v, s6, geo = aniso_geometry(setup, variant)
    h = 1 * mm / n
    f = 5.0 / n                                           # coarser grids: faster growth, coarser far field
    g = 1 + 0.12 * f
    hf = 4 * mm * f
    half_x = v["block_x_mm"] / 2 * mm
    half_y = (2 * v["n_plates_each_side"] - 0.5) * mm
    core = (1.0 * mm, 2.0 * mm)          # whole millimetres = whole multiples of h on every grid (see block_axis)
    x = block_axis(0, geo["Lx"], geo["sx"], half_x, h, core, g, hf)
    y = block_axis(geo["sy"], geo["Ly"], geo["sy"], half_y, h, core, g, hf, one_sided=True)
    for s_, half in (((x - geo["sx"]) / mm, v["block_x_mm"] / 2), ((y - geo["sy"]) / mm, half_y / mm)):
        off = faces_offset(s_, half)
        if off > 1e-6:
            raise RuntimeError(f"block grid error: a half-millimetre boundary is {off:.4f} mm away from the "
                               "midpoint between two nodes (the geometry would not be exact)")
    pt = geo["pt"]
    top = pt + v["block_height_mm"] * mm
    zt0, zt1 = np.asarray(v["recording_tip_z_mm"], float) * mm
    a = geo["a"]
    h0 = h / 9
    nd = max(int(round(a / h0)), 1)
    zz = [(pt + zt0 - 0.5 * mm, pt + zt1 + 0.5 * mm, 0.25 * mm * f), (top - 2 * mm, top + 2 * mm, 1.0 * mm * f)]
    r = geo["ret"]
    if r["type"] == "wire":
        zz.append((r["z0"] - 2 * mm, r["z1"] + 2 * mm, 2 * mm * f))
    z = graded_axis(0, geo["depth"], pt, a / nd, a, g, hf, zones=zz)
    shape = (len(x), len(y), len(z))
    plate, disk, ret, k_top = tank_masks(geo, x, y, z)
    X, Y = np.meshgrid((x - geo["sx"]) / mm, (y - geo["sy"]) / mm, indexing="ij")
    zin = (z > pt + 1e-12) & (z < top)
    sig = np.ones(shape)
    sig[plate] = 1e-8
    if model == "resolved":
        plastic = block_material(v, X, Y)
        sig[plastic[:, :, None] & zin[None, None, :]] = 1e-8
        sigma = sig
    elif model == "tensor":
        box = ((np.abs(X) < v["block_x_mm"] / 2) & (np.abs(Y) < half_y / mm))[:, :, None] & zin[None, None, :]
        comps = []
        for t in tensor:
            c = sig.copy(); c[box] = t
            comps.append(c)
        sigma = tuple(comps)
    elif model == "isotropic":
        sigma = sig
    else:
        raise ValueError(model)
    if verbose:
        print(f"  {variant} {model:9s} h = 1/{n} mm: {shape[0]} x {shape[1]} x {shape[2]} = "
              f"{np.prod(shape):,} nodes; disk {disk.sum()} nodes", flush=True)
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape), floating=[(disk, 0.5)])
    L = st.pop("L")
    cur = current_out_of(L, phi, disk) / 0.5
    del L
    interp = RegularGridInterpolator((x, y, z), phi, method="cubic" if model != "resolved" else "linear")
    zs = np.linspace(pt + zt0, pt + zt1, 11)
    def tips(xy):
        P = np.array([[p[0], geo["sy"] + abs(p[1] - geo["sy"]), zi] for p in xy for zi in zs])
        return interp(P).reshape(len(xy), len(zs)).mean(1)
    dx = np.asarray(v["along_distances_mm"], float) * mm
    dy = np.asarray(v["across_distances_mm"], float) * mm
    along = lambda d: tips(np.column_stack([geo["sx"] + d, np.full(len(d), geo["sy"])]))
    across = lambda d: tips(np.column_stack([np.full(len(d), geo["sx"]), geo["sy"] + d]))
    G_along, G_across = along(dx), across(dy)
    e = pos_sd_mm * mm                                  # sensitivity to the probe position (1 SD)
    dG_along = np.abs(along(dx + e) - along(dx - e)) / 2
    dG_across = np.abs(across(dy + e) - across(dy - e)) / 2
    rx, ry = np.asarray(s6["reference_xy_mm"], float) * mm
    rz = np.linspace(*(np.asarray(s6["reference_exposed_z_mm"], float) * mm), 11)
    G_ref = float(np.mean(interp(np.column_stack([np.full(11, rx), np.full(11, geo["sy"] + abs(ry - geo["sy"])), rz]))))
    return {"G_along": G_along, "G_across": G_across, "dG_pos_along": dG_along, "dG_pos_across": dG_across,
            "G_ref": G_ref, "nodes": int(np.prod(shape)),
            "unknowns": st["unknowns"], "current_check": cur, "h_mm": 1.0 / n,
            "R_disk_to_return_x_sigma": st["floating_potentials"][0]}


def celik(f1, f2, f3, h1, h2, h3, Fs=1.25):
    """Grid uncertainty for three grids with unequal refinement ratios (Celik
    et al. 2008), element by element; f1 on the finest grid (h1 < h2 < h3).
    Where the three values do not converge monotonically, the spread of the
    two finest grids x 3 is used instead (conservative). Returns relative GCI."""
    f1, f2, f3 = (np.asarray(v, float) for v in (f1, f2, f3))
    r21, r32 = h2 / h1, h3 / h2
    out = np.empty_like(f1)
    for i in np.ndindex(f1.shape):
        e21, e32 = f2[i] - f1[i], f3[i] - f2[i]
        if e21 == 0:
            out[i] = 0.0; continue
        s = np.sign(e32 / e21) if e32 != 0 else 0
        if s <= 0:
            out[i] = 3.0 * abs(e21 / f1[i]); continue
        p = 1.0
        for _ in range(200):
            q = np.log(abs((r21 ** p - s) / (r32 ** p - s)))
            pn = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
            if abs(pn - p) < 1e-10:
                break
            p = pn
        p = float(np.clip(p, 0.5, 3.0))
        out[i] = Fs * abs(e21 / f1[i]) / (r21 ** p - 1)
    return out
