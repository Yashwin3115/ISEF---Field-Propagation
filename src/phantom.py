"""Saline phantoms of research plan Parts 6 and 7 (the project's primary
experimental validation): geometry from a setup file the students fill in,
solved with the same verified solver, and the comparison models of Aim 1.

PART 6 -- uniform tank
  Acrylic tank (insulating walls and floor), saline to a fill depth, free
  surface insulating (air). The stimulating wire is cut flush with a printed
  insulating plate -> a metal DISK on an insulating plane (same physics as a
  MaxOne pad, and the case verified exactly in V1). The plate is an
  insulating block on the tank floor, centred under the electrode. Current
  mode: the 100 kOhm series resistor sets the current, so the disk is a
  floating conductor carrying I. The return wire (exposed length L_ret) is
  held at 0 V and is the oscilloscope ground, so every reading is
  phi(recording tip) - phi(return).

PART 7 -- two-conductivity phantom
  An agar sphere (sigma_in) inside a spherical container of saline (sigma_out),
  the container wall insulating. Stimulating tip inside the agar at an offset;
  recording tips at 12 positions over the outer surface.

What is predicted (and committed before measuring)
  The GEOMETRIC FACTOR  G = phi * sigma / I  (units 1/m) at every recording
  position. phi scales exactly with I / sigma for a uniform medium, so the
  measured current and conductivity (measured on the day) turn G into volts:
  V_pred = I_meas * G / sigma_meas. For Part 7 G depends on the ratio
  sigma_in / sigma_out as well, so it is committed at the nominal ratio
  together with its derivative d ln G / d ln(ratio), and corrected linearly
  to the measured ratio -- a rule fixed in advance.

  Comparison models (Aim 1, phantom version):
    point, full space   I / (4 pi sigma r)        (the plan's control)
    point, half space   I / (2 pi sigma r)        (insulating plane, point)
    disk, half space    exact equipotential disk on an infinite insulating
                        plane (no tank, return at infinity) -- isolates the
                        boundary + return path when compared with the full model
    full model          this solver: disk, plate, tank, free surface, return
  With a current source the electrode interface does not change phi at the
  recording tip (the current is measured directly), so "geometry + interface"
  equals "geometry" here; the interface is characterized separately by the
  impedance sweep (step 56, analysis/impedance_fit.py).

Recording tips are thin wires with a 1 mm exposed end: the tip's reading is
the AVERAGE potential along the exposed length (high-impedance input). The
recording wire's own disturbance of the field is not modelled (stated
limitation; it matters only at the closest positions).
"""
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from fd3d import face_mask, solve_general, current_out_of
from grid import graded_axis

mm = 1e-3
um = 1e-6

# ---------------------------------------------------------------------------
# default setup: numbers from research plan v13 where it gives them; the rest
# are marked ASSUMED and must be checked against the built apparatus
# ---------------------------------------------------------------------------
DEFAULT_SETUP = {
    "_comment": "Part 6/7 phantom geometry. Units: mm, S/m, A. Edit to match the "
                "apparatus BEFORE committing predictions. Fields marked ASSUMED are "
                "not in research plan v13 and are guesses.",
    "part6": {
        "tank_inside_mm": [300.0, 200.0, 150.0],          # plan v13 (L x W x H)
        "fill_depth_mm": 100.0,                           # 6.00 L / (300 x 200 mm) = 100 mm
        "stim_xy_mm": [150.0, 100.0],                     # ASSUMED: centre of the floor
        "stim_disk_diameter_mm": 0.50,                    # plan v13: 0.50 mm wire, cut flush
        "plate_mm": [60.0, 60.0, 5.0],                    # ASSUMED: printed plate L x W x thickness
        "return_xy_mm": [290.0, 100.0],                   # ASSUMED: 10 mm from one end wall, mid-width
        "return_exposed_z_mm": [40.0, 60.0],              # plan: 20 mm exposed; ASSUMED height
        "return_wire_diameter_mm": 0.50,
        "recording_direction_xy": [0.0, 1.0],             # ASSUMED: line across the tank (+y)
        "recording_distances_mm": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0],  # plan v13
        "recording_tip_z_mm": [0.5, 1.5],                 # ASSUMED: exposed tip 1 mm, 0.5 mm above plate
        "position_uncertainty_mm": 0.10,                  # ASSUMED: lid hole + calipers
        "tip_height_uncertainty_mm": 0.25,                # ASSUMED
        "sigma_nominal_S_per_m": 1.55,                    # ~9 g/L NaCl near 22 C; measured on the day
        "current_nominal_A": [50e-6, 100e-6],             # plan v13
        "alt_reference_xyz_mm": [20.0, 190.0, 50.0],      # SUGGESTED separate reference wire tip
    },
    "part7": {
        "container_inner_diameter_mm": 150.0,             # plan v13
        "agar_diameter_mm": 80.0,                         # plan v13 (mold)
        "agar_centre_offset_mm": [0.0, 0.0, 0.0],         # measured on the day (step 62)
        "stim_offsets_mm": [[15.0, 0.0, 0.0], [25.0, 0.0, 0.0]],  # ASSUMED: two positions inside agar
        "stim_tip_length_mm": 1.0, "stim_wire_diameter_mm": 0.50,
        "return_position_mm": [-65.0, 0.0, 0.0],          # ASSUMED -- see note
        "return_exposed_length_mm": 5.0,
        "recording_radius_mm": 72.0,                      # ASSUMED: 3 mm inside the wall
        "recording_layout": "icosahedron",                # 12 positions over the surface
        "line_positions_mm": [20, 25, 30, 35, 38, 42, 45, 50, 55, 60, 65, 70],  # SUGGESTED: line
                                                          # along +x crossing the agar boundary,
        "line_z_offset_mm": 5.0,                          # 5 mm above the stimulating tip's level
        "sigma_in_nominal_S_per_m": 2.9,                  # ~18 g/L NaCl agar; measured on the day
        "sigma_out_nominal_S_per_m": 0.80,                # ~4.5 g/L NaCl; measured on the day
        "_note": "Plan v13 step 65 puts the return electrode OUTSIDE the outer container. "
                 "If the container wall is insulating, no current can reach it -- the return "
                 "must be inside the saline. Default here: inside, near the wall opposite the "
                 "stimulating offset. Confirm before committing.",
    },
}


def load_setup(path=None):
    s = json.loads(json.dumps(DEFAULT_SETUP))
    if path is not None:
        user = json.loads(open(path).read())
        for part in ("part6", "part7"):
            s[part].update(user.get(part, {}))
    return s


# ---------------------------------------------------------------------------
# exact / idealized models
# ---------------------------------------------------------------------------

def point_full_space(r):
    """G for a point source in an unbounded medium: 1 / (4 pi r)."""
    return 1.0 / (4 * np.pi * r)


def point_half_space(r):
    """G for a point source on an insulating plane: 1 / (2 pi r)."""
    return 1.0 / (2 * np.pi * r)


def disk_half_space(rho, z, a):
    """G (= phi sigma / I) for an equipotential disk of radius a carrying I on
    an infinite insulating plane, return at infinity (exact):
    phi = I / (2 pi sigma a) * arcsin( 2a / (sqrt((rho-a)^2+z^2) + sqrt((rho+a)^2+z^2)) )."""
    s = np.sqrt((rho - a) ** 2 + z ** 2) + np.sqrt((rho + a) ** 2 + z ** 2)
    return np.arcsin(np.minimum(2 * a / s, 1.0)) / (2 * np.pi * a)


def tip_average(fn, x, y, z0, z1, n=21):
    """Average of fn(x, y, z) along a vertical tip from z0 to z1."""
    zz = np.linspace(z0, z1, n)
    return np.mean([fn(x, y, zi) for zi in zz], axis=0)


# ---------------------------------------------------------------------------
# PART 6: uniform tank
# ---------------------------------------------------------------------------

def tank_geometry(s6):
    g = {k: np.asarray(v, dtype=float) * mm if k.endswith("_mm") else v for k, v in s6.items()}
    Lx, Ly, _ = g["tank_inside_mm"]
    depth = g["fill_depth_mm"]
    sx, sy = g["stim_xy_mm"]
    a = g["stim_disk_diameter_mm"] / 2
    px, py, pt = g["plate_mm"]
    d = np.asarray(s6["recording_direction_xy"], float)
    d = d / np.linalg.norm(d)
    dist = g["recording_distances_mm"]
    rec_xy = np.column_stack([sx + dist * d[0], sy + dist * d[1]])
    ry = g["return_xy_mm"][1]
    # mirror symmetry about y = Ly/2 when the electrode, plate and return wire
    # all sit on that plane: solve half the tank (y >= Ly/2) with the plane
    # insulating, as in step 7; recording points with y < Ly/2 are mirrored
    sym = abs(sy - Ly / 2) < 1e-9 and abs(ry - Ly / 2) < 1e-9
    return dict(Lx=Lx, Ly=Ly, depth=depth, sx=sx, sy=sy, a=a, plate=(px, py, pt),
                rx=g["return_xy_mm"][0], ry=ry,
                rz=g["return_exposed_z_mm"], rw=g["return_wire_diameter_mm"] / 2,
                rec_xy=rec_xy, dist=dist, tip_z=g["recording_tip_z_mm"], direction=d, sym=sym)


def tank_grid(geo, f=1.0):
    """Graded grid: finest across the disk (h = a / (8/f) ... ), fine around the
    return wire and along the recording line, coarse elsewhere. f > 1 coarsens
    every spacing and the growth rate (f = 1, 2, 4 give the GCI sequence)."""
    a = geo["a"]
    n_disk = max(int(round(8 / f)), 2)
    h0 = a / n_disk
    g = 1 + 0.12 * f
    hf = 4 * mm * f
    pt = geo["plate"][2]
    rw = geo["rw"]
    d = geo["direction"]
    dmax = float(np.max(geo["dist"])) + 2 * mm
    wire = lambda c: (c - 3 * rw, c + 3 * rw, max(rw / 3 * f, h0), 1 - 1 / g)
    line = lambda c0, c1: (min(c0, c1) - 1 * mm, max(c0, c1) + 1 * mm, 0.4 * mm * f)
    zx = [wire(geo["rx"])] + ([line(geo["sx"], geo["sx"] + d[0] * dmax)] if abs(d[0]) > 0.1 else [])
    zy = [wire(geo["ry"])] + ([line(geo["sy"], geo["sy"] + abs(d[1]) * dmax)] if abs(d[1]) > 0.1 else [])
    x = graded_axis(0, geo["Lx"], geo["sx"], h0, a, g, hf, zones=zx)
    y = graded_axis(geo["sy"] if geo["sym"] else 0, geo["Ly"], geo["sy"], h0, a, g, hf, zones=zy)
    zr0, zr1 = geo["rz"]
    zt0, zt1 = geo["tip_z"]
    zz = [(pt + zt0 - 0.5 * mm, pt + zt1 + 0.5 * mm, 0.25 * mm * f),
          (zr0 - 2 * mm, zr1 + 2 * mm, 2 * mm * f)]
    # z: plate top at z = pt; nodes exactly on it; below it only the plate block
    z = graded_axis(0, geo["depth"], pt, h0, h0 * n_disk, g, hf, zones=zz)
    return x, y, z


def _lo_hi(a):
    lo, hi = a.copy(), a.copy()
    lo[1:] = 0.5 * (a[1:] + a[:-1]); hi[:-1] = 0.5 * (a[1:] + a[:-1])
    return lo, hi


def vertical_cylinder(x, y, z, cx, cy, radius, z0, z1):
    """Nodes whose control box overlaps a vertical metal cylinder (axis at
    (cx, cy), from z0 to z1). Selecting by overlap, not by node position,
    guarantees a thin wire is never missed on a coarse grid (the lesson of the
    periphery ring, Nov 9, 2026)."""
    xl, xh = _lo_hi(x); yl, yh = _lo_hi(y); zl, zh = _lo_hi(z)
    XL, YL = np.meshgrid(xl, yl, indexing="ij"); XH, YH = np.meshgrid(xh, yh, indexing="ij")
    nx = np.clip(cx, XL, XH); ny = np.clip(cy, YL, YH)
    near = np.hypot(nx - cx, ny - cy) <= radius
    zin = (zh >= z0) & (zl <= z1)
    return near[:, :, None] & zin[None, None, :]


def tank_masks(geo, x, y, z):
    shape = (len(x), len(y), len(z))
    X, Y = np.meshgrid(x, y, indexing="ij")
    px, py, pt = geo["plate"]
    k_top = int(np.argmin(np.abs(z - pt)))
    # insulating plate block: every node strictly inside it, plus its top face
    # outside the disk (the top face nodes INSIDE the disk are the metal)
    in_plate_xy = (np.abs(X - geo["sx"]) <= px / 2) & (np.abs(Y - geo["sy"]) <= py / 2)
    rho = np.hypot(X - geo["sx"], Y - geo["sy"])
    disk_xy = rho <= geo["a"] * (1 + 1e-9)
    plate = np.zeros(shape, bool)
    plate[:, :, :k_top] = in_plate_xy[:, :, None]
    disk = np.zeros(shape, bool)
    disk[:, :, k_top] = disk_xy
    # return wire: nodes whose control box overlaps the exposed cylinder
    zr0, zr1 = geo["rz"]
    ret = vertical_cylinder(x, y, z, geo["rx"], geo["ry"], geo["rw"], zr0, zr1)
    return plate, disk, ret, k_top


def solve_tank(setup, f=1.0, verbose=True):
    """Solve Part 6 at grid factor f. Returns G at every recording position
    (tip-averaged), the disk-to-return resistance x sigma, and the grid."""
    geo = tank_geometry(setup["part6"])
    x, y, z = tank_grid(geo, f)
    shape = (len(x), len(y), len(z))
    plate, disk, ret, k_top = tank_masks(geo, x, y, z)
    if verbose:
        print(f"  grid x{f:g}: {len(x)} x {len(y)} x {len(z)} = {np.prod(shape):,} nodes; "
              f"disk {disk.sum()} nodes, return wire {ret.sum()} nodes")
    sigma = np.ones(shape)                       # sigma = 1 S/m -> phi = G * I
    sigma[plate] = 1e-8                          # insulating block
    I = 0.5 if geo["sym"] else 1.0               # half the tank carries half the current
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape),
                            floating=[(disk, I)])
    L = st.pop("L")
    interp = RegularGridInterpolator((x, y, z), phi, method="cubic")
    pt = geo["plate"][2]
    zt0, zt1 = geo["tip_z"]
    zz = np.linspace(pt + zt0, pt + zt1, 21)
    def G_at(xy, dz=0.0):
        yy = lambda v: geo["sy"] + abs(v - geo["sy"]) if geo["sym"] else v
        P = np.array([[p[0], yy(p[1]), zi + dz] for p in xy for zi in zz])
        return interp(P).reshape(len(xy), len(zz)).mean(1)
    G = G_at(geo["rec_xy"])
    ref = np.asarray(setup["part6"].get("alt_reference_xyz_mm") or [np.nan] * 3, float) * mm
    if np.all(np.isfinite(ref)):
        ry_ = geo["sy"] + abs(ref[1] - geo["sy"]) if geo["sym"] else ref[1]
        G_ref = float(interp([[ref[0], ry_, ref[2]]])[0])
    else:
        G_ref = np.nan
    return {"G": G, "G_alt_reference": G - G_ref, "G_at_alt_reference_point": G_ref, "f": f, "nodes": int(np.prod(shape)), "unknowns": st["unknowns"],
            "symmetry_half": geo["sym"],
            "R_disk_to_return_x_sigma": st["floating_potentials"][0],
            "current_check": current_out_of(L, phi, disk) / I,
            "return_check": -current_out_of(L, phi, ret) / I,
            "G_at": G_at, "geo": geo, "x": x, "y": y, "z": z}


def gci(f_fine, f_med, f_coarse, r=2.0, Fs=1.25):
    """Grid Convergence Index (Roache) for a constant refinement ratio r.
    Returns (gci_fine as relative fraction, observed order p). If the
    differences are not monotone, falls back to Fs=3 with p=1 (conservative)."""
    e21, e32 = f_med - f_fine, f_coarse - f_med
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = e32 / e21
        mono = ratio > 0
        p = np.where(mono, np.log(np.abs(ratio)) / np.log(r), 1.0)
        p = np.clip(p, 0.5, 3.0)
        fs = np.where(mono, Fs, 3.0)
        g = fs * np.abs(e21 / f_fine) / (r ** p - 1)
    return g, p


def idealized_tank(geo):
    """G from the three idealized models at the recording positions (tip-averaged)."""
    pt = geo["plate"][2]
    zt0, zt1 = geo["tip_z"]
    rho = geo["dist"]
    out = {}
    out["point, full space"] = tip_average(lambda xx, yy, zz: point_full_space(np.hypot(rho, zz)),
                                           0, 0, zt0, zt1)
    out["point, half space"] = tip_average(lambda xx, yy, zz: point_half_space(np.hypot(rho, zz)),
                                           0, 0, zt0, zt1)
    out["disk, half space"] = tip_average(lambda xx, yy, zz: disk_half_space(rho, zz, geo["a"]),
                                          0, 0, zt0, zt1)
    return out


# ---------------------------------------------------------------------------
# PART 7: agar sphere in a spherical container
# ---------------------------------------------------------------------------

def icosahedron(R):
    p = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, p, 0], [1, p, 0], [-1, -p, 0], [1, -p, 0],
                  [0, -1, p], [0, 1, p], [0, -1, -p], [0, 1, -p],
                  [p, 0, -1], [p, 0, 1], [-p, 0, -1], [-p, 0, 1]], float)
    return R * v / np.linalg.norm(v, axis=1, keepdims=True)


def solve_sphere(setup, stim_index=0, f=1.0, ratio=None, verbose=True):
    """Solve Part 7 for one stimulating position at grid factor f.
    Coordinates: container centre at the origin. sigma_out = 1 and
    sigma_in = ratio (default: the nominal ratio), so the result is
    G = phi * sigma_out / I."""
    s7 = setup["part7"]
    Rc = s7["container_inner_diameter_mm"] / 2 * mm
    Ra = s7["agar_diameter_mm"] / 2 * mm
    ca = np.asarray(s7["agar_centre_offset_mm"], float) * mm
    st_pos = ca + np.asarray(s7["stim_offsets_mm"][stim_index], float) * mm
    rp = np.asarray(s7["return_position_mm"], float) * mm
    rw = s7["stim_wire_diameter_mm"] / 2 * mm
    tip = s7["stim_tip_length_mm"] * mm
    rl = s7["return_exposed_length_mm"] * mm
    if ratio is None:
        ratio = s7["sigma_in_nominal_S_per_m"] / s7["sigma_out_nominal_S_per_m"]

    n_tip = max(int(round(4 / f)), 1)
    h0 = rw / n_tip
    g = 1 + 0.12 * f
    hf = 4 * mm * f
    def axis(c_stim, c_ret, along):
        half = (rl / 2 if along else rw) + 1 * mm
        zones = [(c_ret - half, c_ret + half, (rw / 2 if not along else 0.5 * mm) * f, 1 - 1 / g)]
        return graded_axis(-Rc - 2 * hf, Rc + 2 * hf, c_stim, h0, h0 * n_tip, g, hf, zones=zones)
    x, y, z = axis(st_pos[0], rp[0], False), axis(st_pos[1], rp[1], False), axis(st_pos[2], rp[2], True)
    shape = (len(x), len(y), len(z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    sigma = np.ones(shape)
    sigma[np.sqrt((X - ca[0]) ** 2 + (Y - ca[1]) ** 2 + (Z - ca[2]) ** 2) <= Ra] = ratio
    sigma[R > Rc] = 1e-8                                   # insulating container wall / outside
    # stimulating tip and return: vertical exposed wire ends (stim floating,
    # carries I; return at 0 V), selected by control-box overlap
    stim = vertical_cylinder(x, y, z, st_pos[0], st_pos[1], rw, st_pos[2] - tip / 2, st_pos[2] + tip / 2)
    ret = vertical_cylinder(x, y, z, rp[0], rp[1], rw, rp[2] - rl / 2, rp[2] + rl / 2)
    del X, Y, Z, R
    if verbose:
        print(f"  sphere grid x{f:g}: {np.prod(shape):,} nodes; tip {stim.sum()} nodes, "
              f"return {ret.sum()} nodes; ratio sigma_in/out {ratio:.3f}")
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape),
                            floating=[(stim, 1.0)])
    st.pop("L")
    rec = icosahedron(s7["recording_radius_mm"] * mm)
    interp = RegularGridInterpolator((x, y, z), phi, method="cubic")
    line_r = np.asarray(s7.get("line_positions_mm") or [], float) * mm
    zz = np.linspace(-0.5 * mm, 0.5 * mm, 11)               # 1 mm vertical tip, centred
    lz = s7.get("line_z_offset_mm", 0.0) * mm + st_pos[2]
    G_line = np.array([np.mean(interp(np.column_stack([np.full(11, r), np.zeros(11), lz + zz])))
                       for r in line_r]) if len(line_r) else np.array([])
    # 1 mm radial tip ending at the recording radius
    u = rec / np.linalg.norm(rec, axis=1, keepdims=True)
    ts = np.linspace(0, 1 * mm, 11)
    P = (rec[:, None, :] + ts[None, :, None] * u[:, None, :]).reshape(-1, 3)
    G = interp(P).reshape(len(rec), len(ts)).mean(1)
    def G_points(P, tip_dirs):
        """Tip-averaged G at points P (n, 3) with 1 mm tips along tip_dirs."""
        Q = (P[:, None, :] + ts[None, :, None] * tip_dirs[:, None, :]).reshape(-1, 3)
        return interp(Q).reshape(len(P), len(ts)).mean(1)
    return {"G": G, "positions_m": rec, "G_line": G_line, "line_x_m": line_r, "G_points": G_points,
            "nodes": int(np.prod(shape)), "ratio": ratio, "stim_position_m": st_pos}
