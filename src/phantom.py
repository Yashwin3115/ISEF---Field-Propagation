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
  held at 0 V. Every position is recorded under two referencing conditions
  (plan v16 steps 53-54):
    A  scope ground at the return:      V_A = phi(tip) - phi(return)
    B  scope ground at a far reference: V_B = phi(tip) - phi(reference wire)
  so V_A - V_B = phi(reference) - phi(return): ONE number, the same at every
  position -- the phantom's measurement of the return-path term.

PART 7 -- two-conductivity phantom (plan v16 steps 65-68)
  An agar sphere (sigma_in) inside a spherical container of saline (sigma_out),
  the container wall insulating. Stimulating tip inside the agar at an offset;
  return wire INSIDE the saline near the wall; recording tips along a straight
  traverse through the agar centre, away from the stimulating electrode,
  crossing the interface perpendicular to it. The headline observable is the
  gradient ratio across the interface (fd_ratio), computed from the same
  points and differences as the measurement, beside the ideal ratio
  sigma_out / sigma_in.

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
    "_comment": "Part 6/7 phantom geometry (research plan v16). Units: mm, S/m, A. Replace "
                "every value marked ASSUMED with the measured one BEFORE committing predictions.",
    "part6": {
        "tank_inside_mm": [300.0, 200.0, 150.0],          # plan (L x W x H); measure (step 44)
        "fill_depth_mm": 100.0,                           # 6.00 L / (300 x 200 mm) = 100 mm
        "stim_xy_mm": [150.0, 100.0],                     # ASSUMED: centre of the floor
        "stim_disk_diameter_mm": 0.50,                    # plan: 0.50 mm wire, cut flush
        "plate_mm": [60.0, 60.0, 5.0],                    # ASSUMED: printed plate L x W x thickness
        "return_xy_mm": [290.0, 100.0],                   # ASSUMED: 10 mm from one end wall, mid-width
        "return_exposed_z_mm": [40.0, 60.0],              # plan: 20 mm exposed; ASSUMED height
        "return_wire_diameter_mm": 0.50,
        "reference_xy_mm": [20.0, 190.0],                 # ASSUMED: far reference wire (step 54)
        "reference_position_uncertainty_mm": 1.0,         # ASSUMED: reference placed by ruler
        "reference_exposed_z_mm": [49.5, 50.5],           # ASSUMED: 1 mm exposed tip at mid-depth
        "recording_direction_xy": [0.0, 1.0],             # ASSUMED: line across the tank (+y)
        "recording_distances_mm": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0],  # plan
        "recording_tip_z_mm": [0.5, 1.5],                 # ASSUMED: exposed tip 1 mm, 0.5 mm above plate
        "position_uncertainty_mm": 0.10,                  # plan v15
        "tip_height_uncertainty_mm": 0.25,                # ASSUMED
        "sigma_nominal_S_per_m": 1.55,                    # ~9 g/L NaCl near 22 C; measured on the day
        "current_nominal_A": [50e-6, 100e-6],             # plan
    },
    "part7": {
        "container_inner_diameter_mm": 150.0,             # plan
        "agar_diameter_mm": 80.0,                         # plan (mold); measured (step 63)
        "agar_centre_offset_mm": [0.0, 0.0, 0.0],         # measured on the day (step 63)
        "stim_offsets_mm": [[15.0, 0.0, 0.0], [25.0, 0.0, 0.0]],  # ASSUMED: two positions in the agar
        "stim_tip_length_mm": 1.0, "stim_wire_diameter_mm": 0.50,
        "return_position_mm": [-66.0, 0.0, -12.0],        # ASSUMED: INSIDE the saline near the wall
                                                          # (step 67), BEYOND the far end of the
                                                          # traverse and just below its line, so the
                                                          # current crosses the interface along it.
                                                          # Off to the side (e.g. [0, 65, 0]) the
                                                          # traverse sees ~15x less signal.
        "return_exposed_length_mm": 5.0,                  # ASSUMED
        "traverse_direction": [-1.0, 0.0, 0.0],           # through the agar centre, AWAY from the
                                                          # stimulating electrode (step 68)
        "traverse_distances_mm": [5, 15, 25, 30, 35, 37.5, 42.5, 45, 50, 55, 60, 65],
                                                          # from the agar centre; >= 4 inside, >= 4 in
                                                          # saline, rest within 5 mm of the interface.
                                                          # Nearest-point spacing: runs/traverse_design.py
        "position_uncertainty_mm": 0.10,                  # plan v15 (step 66)
        "voltage_noise_V": 5e-6,                          # ASSUMED residual noise after averaging 20
                                                          # traces; replace with the session noise floor
        "sigma_in_nominal_S_per_m": 2.9,                  # ~18 g/L NaCl agar; measured (step 62)
        "sigma_out_nominal_S_per_m": 0.80,                # ~4.5 g/L NaCl; measured (step 64)
        "current_nominal_A": [50e-6, 100e-6],
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
    # far reference wire (condition B): a high-impedance wire, so it reads the
    # AVERAGE potential along its exposed tip and does not disturb the field
    s6 = setup["part6"]
    rx_, ry_ = np.asarray(s6["reference_xy_mm"], float) * mm
    rz0, rz1 = np.asarray(s6["reference_exposed_z_mm"], float) * mm
    if geo["sym"]:
        ry_ = geo["sy"] + abs(ry_ - geo["sy"])
    zr = np.linspace(rz0, rz1, 11)
    G_ref = float(np.mean(interp(np.column_stack([np.full(11, rx_), np.full(11, ry_), zr]))))
    def G_point(P):
        """G at arbitrary points (n, 3), mirrored into the solved half if needed."""
        P = np.array(P, float).reshape(-1, 3)
        if geo["sym"]:
            P[:, 1] = geo["sy"] + np.abs(P[:, 1] - geo["sy"])
        return interp(P)
    def G_reference(xy_m, z0_m, z1_m):
        zz_ = np.linspace(z0_m, z1_m, 11)
        return float(np.mean(G_point(np.column_stack([np.full(11, xy_m[0]), np.full(11, xy_m[1]), zz_]))))
    return {"G": G, "G_condition_A": G, "G_condition_B": G - G_ref, "G_reference_wire": G_ref,
            "G_point": G_point, "G_reference": G_reference,
            "f": f, "nodes": int(np.prod(shape)), "unknowns": st["unknowns"],
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

def fd_ratio(r, phi, r_iface):
    """Gradient ratio across the interface FROM DISCRETE POINTS, exactly as it
    will be computed from the measurements: slope between the two traverse
    points nearest the interface on the inside, divided by the slope between
    the two nearest on the outside. r: distances from the agar centre along
    the traverse (sorted); phi: potentials there. For an exact interface the
    limit is sigma_out / sigma_in (normal current continuous)."""
    r = np.asarray(r, float); phi = np.asarray(phi, float)
    i_in = np.flatnonzero(r < r_iface); i_out = np.flatnonzero(r > r_iface)
    a, b = i_in[-2], i_in[-1]
    c, d = i_out[0], i_out[1]
    s_in = (phi[b] - phi[a]) / (r[b] - r[a])
    s_out = (phi[d] - phi[c]) / (r[d] - r[c])
    return s_in / s_out, (a, b, c, d)


def sphere_geometry(s7, stim_index):
    Rc = s7["container_inner_diameter_mm"] / 2 * mm
    Ra = s7["agar_diameter_mm"] / 2 * mm
    ca = np.asarray(s7["agar_centre_offset_mm"], float) * mm
    st_pos = ca + np.asarray(s7["stim_offsets_mm"][stim_index], float) * mm
    rp = np.asarray(s7["return_position_mm"], float) * mm
    u = np.asarray(s7["traverse_direction"], float); u = u / np.linalg.norm(u)
    return Rc, Ra, ca, st_pos, rp, u


def solve_sphere(setup, stim_index=0, f=1.0, ratio=None, verbose=True):
    """Solve Part 7 (plan v15 steps 65-68) for one stimulating position at grid
    factor f. Container centre at the origin; sigma_out = 1 and sigma_in =
    ratio (default: the nominal ratio), so the result is G = phi sigma_out / I.

    Geometry: agar sphere (sigma_in) inside the saline (sigma_out), container
    wall insulating; stimulating wire tip inside the agar (floating, carries
    I); return wire INSIDE the saline (0 V; the oscilloscope ground). The
    traverse runs through the agar centre along `traverse_direction`, so it
    crosses the interface perpendicular to the surface. The grid is refined
    along the traverse and most finely where it crosses the interface."""
    s7 = setup["part7"]
    Rc, Ra, ca, st_pos, rp, u = sphere_geometry(s7, stim_index)
    rw = s7["stim_wire_diameter_mm"] / 2 * mm
    tip = s7["stim_tip_length_mm"] * mm
    rl = s7["return_exposed_length_mm"] * mm
    if ratio is None:
        ratio = s7["sigma_in_nominal_S_per_m"] / s7["sigma_out_nominal_S_per_m"]
    cross = ca + Ra * u                                    # where the traverse meets the interface
    end = ca + (Rc - 2 * mm) * u
    n_tip = max(int(round(4 / f)), 1)
    h0 = rw / n_tip
    g = 1 + 0.12 * f
    hf = 4 * mm * f
    def axis(k, along):
        # the return wire sets the absolute level of every reading (its own
        # spreading resistance), so resolve it well: across it rw/3, along it rw
        zones = [(rp[k] - (rl / 2 if along else rw) - 1 * mm, rp[k] + (rl / 2 if along else rw) + 1 * mm,
                  (rw if along else rw / 3) * f, 1 - 1 / g)]
        lo_t, hi_t = sorted([ca[k], end[k]])
        if hi_t - lo_t > 1 * mm:                           # traverse runs along this axis
            zones.append((lo_t - 1 * mm, hi_t + 1 * mm, 0.5 * mm * f))
            zones.append((cross[k] - 1.5 * mm, cross[k] + 1.5 * mm, 0.15 * mm * f, 1 - 1 / g))
        else:                                              # across the traverse: keep it resolved
            zones.append((cross[k] - 2 * mm, cross[k] + 2 * mm, 0.5 * mm * f))
        return graded_axis(-Rc - 2 * hf, Rc + 2 * hf, st_pos[k], h0, h0 * n_tip, g, hf, zones=zones)
    # mirror symmetry about the plane y = 0 when the stimulating tip, the
    # return, the agar centre and the traverse all lie in it: solve y >= 0
    # with the plane insulating (as for the tank), half the current
    sym = (abs(st_pos[1]) < 1e-9 and abs(rp[1]) < 1e-9 and abs(ca[1]) < 1e-9 and abs(u[1]) < 1e-12)
    x, y, z = axis(0, False), axis(1, False), axis(2, True)
    if sym:
        y = y[y >= -1e-12]
        y[0] = 0.0
    shape = (len(x), len(y), len(z))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    sigma = np.ones(shape)
    sigma[np.sqrt((X - ca[0]) ** 2 + (Y - ca[1]) ** 2 + (Z - ca[2]) ** 2) <= Ra] = ratio
    sigma[R > Rc] = 1e-8                                   # insulating container wall / outside
    stim = vertical_cylinder(x, y, z, st_pos[0], st_pos[1], rw, st_pos[2] - tip / 2, st_pos[2] + tip / 2)
    ret = vertical_cylinder(x, y, z, rp[0], rp[1], rw, rp[2] - rl / 2, rp[2] + rl / 2)
    del X, Y, Z, R
    if np.hypot(rp[0], rp[1]) > Rc or np.linalg.norm(rp) > Rc:
        raise ValueError("return electrode outside the container: no current could reach it")
    if verbose:
        print(f"  sphere grid x{f:g}: {np.prod(shape):,} nodes; tip {stim.sum()} nodes, "
              f"return {ret.sum()} nodes; sigma_in/out {ratio:.3f}")
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape),
                            floating=[(stim, 0.5 if sym else 1.0)])
    st.pop("L")
    interp = RegularGridInterpolator((x, y, z), phi, method="cubic")
    zz = np.linspace(-0.5 * mm, 0.5 * mm, 11)              # 1 mm vertical recording tip, centred
    def G_traverse(dist_m):
        """Tip-averaged G at distances (m) from the agar centre along the traverse."""
        dist_m = np.atleast_1d(np.asarray(dist_m, float))
        P = ca[None, :] + dist_m[:, None] * u[None, :]
        Q = (P[:, None, :] + np.array([0, 0, 1.0])[None, None, :] * zz[None, :, None]).reshape(-1, 3)
        return interp(Q).reshape(len(P), len(zz)).mean(1)
    d = np.sort(np.asarray(s7["traverse_distances_mm"], float)) * mm
    G = G_traverse(d)
    return {"G": G, "distances_m": d, "G_traverse": G_traverse, "interface_m": Ra,
            "nodes": int(np.prod(shape)), "ratio": ratio, "stim_position_m": st_pos,
            "symmetry_half": bool(sym)}


def design_spacing(G_traverse, r_iface, ratio_ideal, sigma_out, current_A, noise_V, pos_unc_m,
                   spacings_m=None, max_outer_m=5 * mm):
    """Plan v15 step 66: choose the spacing d of the points nearest the
    interface (at r_iface -/+ d and -/+ 2d) by error propagation, BEFORE the
    session. For each d:
      gap    (finite-difference ratio / ideal ratio) - 1   (closer points: smaller)
      U_pos  ratio change for +-pos_unc on each of the 4 points (linear, RSS)
      U_noise  ratio change for +-noise_V on each of the 4 readings (RSS)
    combined = sqrt(gap^2 + U_pos^2 + U_noise^2); returns the table and the
    d that minimizes it. Potentials in volts: V = I G / sigma_out."""
    if spacings_m is None:
        spacings_m = np.arange(0.5, 6.01, 0.25) * mm
    rows = []
    for d in spacings_m:
        r = np.array([r_iface - 2 * d, r_iface - d, r_iface + d, r_iface + 2 * d])
        V = current_A * G_traverse(r) / sigma_out
        R0 = ((V[1] - V[0]) / d) / ((V[3] - V[2]) / d)
        # position: shift each point by +-pos_unc (potential follows the local slope)
        dV = current_A * (G_traverse(r + 1e-5) - G_traverse(r - 1e-5)) / (2e-5) / sigma_out
        def ratio_of(Vv, rr):
            return ((Vv[1] - Vv[0]) / (rr[1] - rr[0])) / ((Vv[3] - Vv[2]) / (rr[3] - rr[2]))
        up = []
        for k in range(4):
            rr = r.copy(); rr_true = r.copy(); rr_true[k] += pos_unc_m
            Vt = V.copy(); Vt[k] += dV[k] * pos_unc_m          # measured at a slightly wrong place
            up.append(ratio_of(Vt, rr) / R0 - 1)
        un = []
        for k in range(4):
            Vn = V.copy(); Vn[k] += noise_V
            un.append(ratio_of(Vn, r) / R0 - 1)
        U_pos = float(np.sqrt(np.sum(np.square(up))))
        U_noise = float(np.sqrt(np.sum(np.square(un))))
        gap = float(R0 / ratio_ideal - 1)
        rows.append({"spacing_mm": float(d / mm), "fd_ratio": float(R0), "gap_rel": gap,
                     "U_pos_rel": U_pos, "U_noise_rel": U_noise,
                     "combined_rel": float(np.sqrt(gap ** 2 + U_pos ** 2 + U_noise ** 2)),
                     "dV_inside_pair_V": float(abs(V[1] - V[0])), "dV_outside_pair_V": float(abs(V[3] - V[2]))})
    # plan step 68: the near-interface points lie within 5 mm of the interface,
    # so the outer point of each pair (at 2d) must be <= max_outer_m away
    allowed = [q for q in rows if 2 * q["spacing_mm"] * mm <= max_outer_m + 1e-12]
    best = min(allowed, key=lambda q: q["combined_rel"])
    best_unconstrained = min(rows, key=lambda q: q["combined_rel"])
    best = dict(best, best_without_5mm_rule_mm=best_unconstrained["spacing_mm"],
                combined_without_5mm_rule_rel=best_unconstrained["combined_rel"])
    return rows, best
