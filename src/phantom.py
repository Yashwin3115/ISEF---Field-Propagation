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

EXPERIMENTS (Nov 15, 2026: the phantom is the project's primary validation)
  The setup file lists named Part 6 layouts under "experiments"; each one is
  the baseline tank with some fields replaced, and each is predicted blind:
    E1_distance     disk flush in the plate, return wire at the far wall:
                    potential against distance, full model vs point source
    E2_no_plate     the SAME wire end with no plate (on a thin insulated post
                    in open saline): the plate should about double the reading
    E3_return_top   return = a metal sheet above the electrode
    E3_return_ring  return = a ring around the electrode, on the plate
                    (with E1 these are the array's three return paths at
                    phantom scale)
  Supports: "plate" (insulating block) or "post" (insulated wire/tube).
  Returns:  "wire" (vertical exposed wire), "plate" (horizontal metal sheet),
            "ring" (horizontal wire or strip ring).

PREDICTION BANDS (tank_bands, sphere_bands)
  Every committed number is a band: a Monte Carlo over the measured inputs
  (each probe's position and height, the electrode diameter, the reference
  wire's position) through the fine-grid solution, with the grid uncertainty
  (GCI) drawn in the same samples. Conductivity, temperature and current
  enter as V = I G / sigma; their stated uncertainties give the band in
  mV per uA, and the comparison recomputes that part from the values
  measured on the day.
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
    "_comment": "Phantom geometry and instruments (research plan v16 + Nov 15 experiments). Units: "
                "mm, S/m, A, V. Replace every value marked ASSUMED with the measured one BEFORE "
                "committing predictions.",
    "instrument": {
        "_comment": "NOT YET CONFIRMED (Nov 15): set these from the school's instruments.",
        "scope_bits": 8,                                  # ASSUMED
        "scope_full_scale_divisions": 8.0,                # ASSUMED (some scopes: 10)
        "scope_min_volts_per_div": 0.002,                 # ASSUMED: smallest V/div
        "scope_channels": 2,                              # ASSUMED
        "scope_single_sample_noise_V": 150e-6,            # ASSUMED; measure: runs/scope_noise.py
        "scope_samples_in_plateau": 500,                  # ASSUMED: samples in the middle half of a phase
        "generator_channels": 1,                          # ASSUMED (2 needed for physical spikes)
        "generator_max_amplitude_V": 10.0,                # ASSUMED: peak, into the series resistor
        "series_resistor_ohm": 100e3,                     # plan
    },
    "pulse": {
        "_comment": "Stimulus settings (ASSUMED until the generator is confirmed).",
        "phase_widths_us": [100.0, 200.0, 500.0],         # plan: three pulse widths
        "interphase_us": 0.0,
        "repetition_Hz": 100.0,                           # ASSUMED
        "generator_amplitude_V": 10.0,                    # ASSUMED: 10 V / 100 kOhm = 100 uA
        "generator_rise_time_s": 0.0,                     # from the generator's manual, if stated
    },
    "uncertainties": {
        "_comment": "Standard uncertainties (1 SD) of the inputs, for the Monte Carlo bands.",
        "probe_position_sd_mm": 0.10,                     # plan: +-0.1 mm, taken as 1 SD, along and across
        "probe_height_sd_mm": 0.25,                       # ASSUMED
        "electrode_diameter_rel_sd": 0.02,                # 0.50 +- 0.01 mm (caliper)
        "reference_position_sd_mm": 1.0,                  # ASSUMED: placed by ruler
        "sigma_meter_rel_sd": 0.01,                       # ASSUMED: conductivity meter
        "temperature_sd_C": 0.5,                          # ASSUMED: between the sigma reading and the trace
        "sigma_temp_coeff_per_C": 0.02,                   # NaCl solutions, about 2 % per C near 20-25 C
        "current_rel_sd": 0.01,                           # 1 % resistor; measure it with a multimeter
        "scope_gain_rel_sd": 0.015,                       # ASSUMED: 3 % vertical accuracy as a 95 % bound
        "mc_samples": 4000, "mc_seed": 2026,
    },
    "part6": {
        "tank_inside_mm": [300.0, 200.0, 150.0],          # plan (L x W x H); measure (step 44)
        "fill_depth_mm": 100.0,                           # 6.00 L / (300 x 200 mm) = 100 mm
        "stim_xy_mm": [150.0, 100.0],                     # ASSUMED: centre of the floor
        "stim_disk_diameter_mm": 0.50,                    # plan: 0.50 mm wire, cut flush
        "plate_mm": [80.0, 80.0, 5.0],                    # ASSUMED: printed plate L x W x thickness
        "return_xy_mm": [290.0, 100.0],                   # ASSUMED: 10 mm from one end wall, mid-width
        "return_exposed_z_mm": [40.0, 60.0],              # plan: 20 mm exposed; ASSUMED height
        "return_wire_diameter_mm": 0.50,
        "reference_xy_mm": [20.0, 190.0],                 # ASSUMED: far reference wire (step 54)
        "reference_position_uncertainty_mm": 1.0,         # ASSUMED: reference placed by ruler
        "reference_exposed_z_mm": [49.5, 50.5],           # ASSUMED: 1 mm exposed tip at mid-depth
        "recording_direction_xy": [0.0, 1.0],             # ASSUMED: line across the tank (+y)
        "recording_distances_mm": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 45.0, 60.0],
                                                          # plan: lid holes 1-20 mm; 25, 45 and 60 mm are
                                                          # PROPOSED extra holes (Nov 15), where the tank
                                                          # walls separate the full model from a point source
        "recording_tip_z_mm": [0.5, 1.5],                 # ASSUMED: exposed tip 1 mm, 0.5 mm above plate
        "position_uncertainty_mm": 0.10,                  # plan v15
        "tip_height_uncertainty_mm": 0.25,                # ASSUMED
        "sigma_nominal_S_per_m": 1.55,                    # ~9 g/L NaCl near 22 C; measured on the day
        "current_nominal_A": [50e-6, 100e-6],             # plan
    },
    "experiments": {
        "_comment": "Part 6 layouts. Each = part6 with the listed fields replaced. Delete a layout "
                    "that will not be built. Geometry here is ASSUMED until measured.",
        "E1_distance": {
            "description": "Disk flush in the insulating plate, return wire at the far wall",
            "overrides": {}},
        "E2_no_plate": {
            "description": "Same wire end, no plate: on a thin insulated post in open saline",
            "overrides": {"support": {"type": "post", "diameter_mm": 1.0, "height_mm": 50.0}}},
                                                          # ASSUMED: post = the wire in its own insulation
                                                          # (outer diameter 1.0 mm), tip 50 mm above the floor
        "E3_return_top": {
            "description": "Return = metal sheet above the electrode (the array's top-plate return)",
            "overrides": {"return": {"type": "plate", "size_mm": [100.0, 100.0],
                                     "centre_xy_mm": [150.0, 100.0], "height_above_stim_mm": 20.0}}},
                                                          # ASSUMED; height null = at the saline surface
        "E3_return_ring": {
            "description": "Return = ring on the plate around the electrode (the array's periphery return)",
            "overrides": {"return": {"type": "ring", "centre_xy_mm": [150.0, 100.0],
                                     "radius_mm": 30.0, "width_mm": 0.5}}},
                                                          # ASSUMED: 0.5 mm wire ring; a flat strip is
                                                          # width_mm = strip width, radius_mm = its mid-radius
    },
    "saline_concentrations": {
        "_comment": "Two concentrations (experiment e): readings x sigma / I must collapse. "
                    "Nominal conductivities; the measured values go in the sessions CSV.",
        "nominal_sigma_S_per_m": [1.55, 0.80],
    },
    "anisotropic": None,                                  # filled from aniso_phantom.ANISO_DEFAULT by load_setup
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
        for part in ("part6", "part7", "instrument", "pulse", "uncertainties", "saline_concentrations"):
            s[part].update(user.get(part, {}))
        if "experiments" in user:                         # the user's list replaces the default list
            s["experiments"] = user["experiments"]
    from aniso_phantom import ANISO_DEFAULT              # (imported here: aniso_phantom imports this module)
    an = json.loads(json.dumps(ANISO_DEFAULT))
    if path is not None:
        an.update(user.get("anisotropic") or {})
    s["anisotropic"] = an
    return s


def experiment_names(setup):
    return [k for k in setup.get("experiments", {}) if not k.startswith("_")]


def layout_setup(setup, name=None):
    """The part6 block for one experiment: the baseline with that experiment's
    overrides applied. name None -> the baseline itself."""
    s6 = json.loads(json.dumps(setup["part6"]))
    if name is not None:
        s6.update(setup["experiments"][name].get("overrides", {}))
    return s6


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
    d = np.asarray(s6["recording_direction_xy"], float)
    d = d / np.linalg.norm(d)
    dist = g["recording_distances_mm"]
    rec_xy = np.column_stack([sx + dist * d[0], sy + dist * d[1]])
    # --- what the disk sits in: an insulating plate (block) or a thin insulated post
    sup = s6.get("support")
    if sup is None or sup.get("type", "plate") == "plate":
        size = sup["size_mm"] if sup else s6["plate_mm"]
        support = {"type": "plate", "px": size[0] * mm, "py": size[1] * mm}
        pt = size[2] * mm
    elif sup["type"] == "post":
        support = {"type": "post", "radius": sup["diameter_mm"] / 2 * mm}
        pt = sup["height_mm"] * mm
        if support["radius"] < a:
            raise ValueError("post diameter is smaller than the electrode")
    else:
        raise ValueError(f"support type must be 'plate' or 'post', got {sup['type']!r}")
    # --- the return electrode
    rt = s6.get("return")
    if rt is None or rt.get("type", "wire") == "wire":
        xy = rt["xy_mm"] if rt else s6["return_xy_mm"]
        zz = rt["exposed_z_mm"] if rt else s6["return_exposed_z_mm"]
        wd = rt["wire_diameter_mm"] if rt else s6["return_wire_diameter_mm"]
        ret = {"type": "wire", "x": xy[0] * mm, "y": xy[1] * mm, "z0": zz[0] * mm, "z1": zz[1] * mm,
               "rw": wd / 2 * mm}
        ret_sym_y = ret["y"]
    elif rt["type"] == "plate":
        h = rt.get("height_above_stim_mm")
        z = depth if h is None else pt + h * mm
        if z > depth + 1e-12:
            raise ValueError("the return sheet is above the saline surface")
        ret = {"type": "plate", "cx": rt["centre_xy_mm"][0] * mm, "cy": rt["centre_xy_mm"][1] * mm,
               "sx": rt["size_mm"][0] * mm, "sy": rt["size_mm"][1] * mm, "z": z}
        ret_sym_y = ret["cy"]
    elif rt["type"] == "ring":
        ret = {"type": "ring", "cx": rt["centre_xy_mm"][0] * mm, "cy": rt["centre_xy_mm"][1] * mm,
               "R": rt["radius_mm"] * mm, "hw": rt["width_mm"] / 2 * mm,
               "z": pt + rt.get("height_above_stim_mm", 0.0) * mm}
        ret_sym_y = ret["cy"]
    else:
        raise ValueError(f"return type must be 'wire', 'plate' or 'ring', got {rt['type']!r}")
    # mirror symmetry about y = Ly/2 when the electrode, its support and the
    # return all sit on that plane: solve half the tank (y >= Ly/2) with the
    # plane insulating, as in step 7; recording points with y < Ly/2 are mirrored
    sym = abs(sy - Ly / 2) < 1e-9 and abs(ret_sym_y - Ly / 2) < 1e-9
    return dict(Lx=Lx, Ly=Ly, depth=depth, sx=sx, sy=sy, a=a, pt=pt, support=support, ret=ret,
                rec_xy=rec_xy, dist=dist, tip_z=g["recording_tip_z_mm"], direction=d, sym=sym)


def tank_grid(geo, f=1.0):
    """Graded grid: finest across the disk (h = a / (8/f) ... ), fine around the
    return and along the recording line, coarse elsewhere. f > 1 coarsens
    every spacing and the growth rate (f = 1, 2, 4 give the GCI sequence)."""
    a = geo["a"]
    n_disk = max(int(round(8 / f)), 2)
    h0 = a / n_disk
    g = 1 + 0.12 * f
    hf = 4 * mm * f
    pt = geo["pt"]
    d = geo["direction"]
    dmax = float(np.max(geo["dist"])) + 2 * mm
    # along the recording line: 0.4 mm out to 27 mm (where the potential changes
    # fast), 1 mm beyond that (the far points see a slowly varying field)
    dnear = min(dmax, 27 * mm)
    def line(c0, u):
        zs = [(min(c0, c0 + u * dnear) - 1 * mm, max(c0, c0 + u * dnear) + 1 * mm, 0.4 * mm * f)]
        if dmax > dnear:
            zs.append((min(c0, c0 + u * dmax) - 1 * mm, max(c0, c0 + u * dmax) + 1 * mm, 1.0 * mm * f))
        return zs
    zx = line(geo["sx"], d[0]) if abs(d[0]) > 0.1 else []
    zy = line(geo["sy"], abs(d[1]) if geo["sym"] else d[1]) if abs(d[1]) > 0.1 else []
    zt0, zt1 = geo["tip_z"]
    zz = [(pt + zt0 - 0.5 * mm, pt + zt1 + 0.5 * mm, 0.25 * mm * f)]
    r = geo["ret"]
    if r["type"] == "wire":
        rw = r["rw"]
        wire = lambda c: (c - 3 * rw, c + 3 * rw, max(rw / 3 * f, h0), 1 - 1 / g)
        zx.append(wire(r["x"])); zy.append(wire(r["y"]))
        zz.append((r["z0"] - 2 * mm, r["z1"] + 2 * mm, 2 * mm * f))
    elif r["type"] == "plate":                       # a sheet: resolve its height and its edges
        for zl, c, half in ((zx, r["cx"], r["sx"] / 2), (zy, r["cy"], r["sy"] / 2)):
            zl.append((c - half - 2 * mm, c + half + 2 * mm, 2 * mm * f))
            for e in (c - half, c + half):
                zl.append((e - 1 * mm, e + 1 * mm, 1.0 * mm * f, 1 - 1 / g))
        zz.append((r["z"] - 1 * mm, r["z"] + 1 * mm, 0.5 * mm * f, 1 - 1 / g))
    else:                                            # ring: keep the whole disc it encloses fine
        hc = max(min(0.5 * mm, 2 * r["hw"]) * f, h0)
        zx.append((r["cx"] - r["R"] - 2 * mm, r["cx"] + r["R"] + 2 * mm, hc))
        zy.append((r["cy"] - r["R"] - 2 * mm, r["cy"] + r["R"] + 2 * mm, hc))
        zz.append((r["z"] - 0.5 * mm, r["z"] + 1 * mm, 0.25 * mm * f))
    x = graded_axis(0, geo["Lx"], geo["sx"], h0, a, g, hf, zones=zx)
    y = graded_axis(geo["sy"] if geo["sym"] else 0, geo["Ly"], geo["sy"], h0, a, g, hf, zones=zy)
    # z: disk plane at z = pt; nodes exactly on it; below it only the support
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


def horizontal_ring(x, y, z, cx, cy, R, half_width, z0, z1):
    """Nodes whose control box overlaps a horizontal ring: the annulus
    R - half_width <= rho <= R + half_width, between heights z0 and z1.
    A box overlaps the annulus when its nearest point to the centre is inside
    the outer radius and its farthest point is outside the inner radius, so
    the selected nodes always form a closed ring, however coarse the grid."""
    xl, xh = _lo_hi(x); yl, yh = _lo_hi(y); zl, zh = _lo_hi(z)
    XL, YL = np.meshgrid(xl, yl, indexing="ij"); XH, YH = np.meshgrid(xh, yh, indexing="ij")
    nearest = np.hypot(np.clip(cx, XL, XH) - cx, np.clip(cy, YL, YH) - cy)
    farthest = np.hypot(np.maximum(np.abs(XL - cx), np.abs(XH - cx)),
                        np.maximum(np.abs(YL - cy), np.abs(YH - cy)))
    ann = (nearest <= R + half_width) & (farthest >= R - half_width)
    zin = (zh >= z0) & (zl <= z1)
    return ann[:, :, None] & zin[None, None, :]


def tank_masks(geo, x, y, z):
    shape = (len(x), len(y), len(z))
    X, Y = np.meshgrid(x, y, indexing="ij")
    pt = geo["pt"]
    k_top = int(np.argmin(np.abs(z - pt)))
    # insulating support: every node strictly below the disk plane inside it;
    # the nodes ON the plane inside the disk are the metal, the rest of the
    # plane is saline resting on the insulator
    rho = np.hypot(X - geo["sx"], Y - geo["sy"])
    sp = geo["support"]
    if sp["type"] == "plate":
        in_xy = (np.abs(X - geo["sx"]) <= sp["px"] / 2) & (np.abs(Y - geo["sy"]) <= sp["py"] / 2)
    else:
        in_xy = rho <= sp["radius"] * (1 + 1e-9)
    disk_xy = rho <= geo["a"] * (1 + 1e-9)
    plate = np.zeros(shape, bool)
    plate[:, :, :k_top] = in_xy[:, :, None]
    disk = np.zeros(shape, bool)
    disk[:, :, k_top] = disk_xy
    r = geo["ret"]
    if r["type"] == "wire":                           # nodes whose control box overlaps the cylinder
        ret = vertical_cylinder(x, y, z, r["x"], r["y"], r["rw"], r["z0"], r["z1"])
    elif r["type"] == "plate":                        # one layer of nodes: a thin metal sheet
        k = int(np.argmin(np.abs(z - r["z"])))
        ret = np.zeros(shape, bool)
        ret[:, :, k] = (np.abs(X - r["cx"]) <= r["sx"] / 2) & (np.abs(Y - r["cy"]) <= r["sy"] / 2)
    else:
        k = int(np.argmin(np.abs(z - r["z"])))
        ret = horizontal_ring(x, y, z, r["cx"], r["cy"], r["R"], r["hw"], z[k] - 1e-12, z[k] + 1e-12)
    ret &= ~plate
    if np.any(ret & disk):
        raise ValueError("the return electrode touches the stimulating disk")
    return plate, disk, ret, k_top


def solve_tank(setup, f=1.0, verbose=True, layout=None, s6=None):
    """Solve one Part 6 layout at grid factor f (layout: a name under
    "experiments", or None for the baseline; s6 overrides both). Returns G at
    every recording position (tip-averaged) under both referencing conditions,
    the disk-to-return resistance x sigma, and functions to evaluate G anywhere."""
    if s6 is None:
        s6 = layout_setup(setup, layout)
    geo = tank_geometry(s6)
    x, y, z = tank_grid(geo, f)
    shape = (len(x), len(y), len(z))
    plate, disk, ret, k_top = tank_masks(geo, x, y, z)
    if verbose:
        print(f"  grid x{f:g}: {len(x)} x {len(y)} x {len(z)} = {np.prod(shape):,} nodes; "
              f"disk {disk.sum()} nodes, return ({geo['ret']['type']}) {ret.sum()} nodes")
    sigma = np.ones(shape)                       # sigma = 1 S/m -> phi = G * I
    sigma[plate] = 1e-8                          # insulating support
    I = 0.5 if geo["sym"] else 1.0               # half the tank carries half the current
    phi, st = solve_general(x, y, z, sigma, fixed=ret, phi_fixed=np.zeros(shape),
                            floating=[(disk, I)])
    L = st.pop("L")
    cur = current_out_of(L, phi, disk) / I
    back = -current_out_of(L, phi, ret) / I
    del L
    interp = RegularGridInterpolator((x, y, z), phi, method="cubic")
    pt = geo["pt"]
    zt0, zt1 = geo["tip_z"]
    zz = np.linspace(pt + zt0, pt + zt1, 21)
    def G_point(P):
        """G at arbitrary points (n, 3), mirrored into the solved half if needed."""
        P = np.array(P, float).reshape(-1, 3)
        if geo["sym"]:
            P[:, 1] = geo["sy"] + np.abs(P[:, 1] - geo["sy"])
        P[:, 0] = np.clip(P[:, 0], x[0], x[-1]); P[:, 1] = np.clip(P[:, 1], y[0], y[-1])
        P[:, 2] = np.clip(P[:, 2], z[0], z[-1])
        return interp(P)
    def G_at(xy, dz=0.0):
        """Tip-averaged G at (n, 2) positions; dz: height offset, one value or one per position."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        dzv = np.broadcast_to(np.asarray(dz, float), (len(xy),))
        P = np.empty((len(xy), len(zz), 3))
        P[:, :, 0] = xy[:, None, 0]; P[:, :, 1] = xy[:, None, 1]
        P[:, :, 2] = zz[None, :] + dzv[:, None]
        return G_point(P.reshape(-1, 3)).reshape(len(xy), len(zz)).mean(1)
    def G_reference(xy_m, z0_m, z1_m):
        """Far reference wire: a high-impedance wire reads the AVERAGE potential
        along its exposed tip. xy_m may be (2,) or (n, 2)."""
        xy = np.asarray(xy_m, float).reshape(-1, 2)
        z0v = np.broadcast_to(np.asarray(z0_m, float), (len(xy),))
        z1v = np.broadcast_to(np.asarray(z1_m, float), (len(xy),))
        w = np.linspace(0, 1, 11)
        P = np.empty((len(xy), 11, 3))
        P[:, :, 0] = xy[:, None, 0]; P[:, :, 1] = xy[:, None, 1]
        P[:, :, 2] = z0v[:, None] + (z1v - z0v)[:, None] * w[None, :]
        v = G_point(P.reshape(-1, 3)).reshape(len(xy), 11).mean(1)
        return float(v[0]) if np.ndim(xy_m) == 1 else v
    G = G_at(geo["rec_xy"])
    ref_xy = np.asarray(s6["reference_xy_mm"], float) * mm
    rz0, rz1 = np.asarray(s6["reference_exposed_z_mm"], float) * mm
    G_ref = G_reference(ref_xy, rz0, rz1)
    return {"G": G, "G_condition_A": G, "G_condition_B": G - G_ref, "G_reference_wire": G_ref,
            "G_point": G_point, "G_reference": G_reference, "ref_xy": ref_xy, "ref_z": (rz0, rz1),
            "f": f, "nodes": int(np.prod(shape)), "unknowns": st["unknowns"],
            "symmetry_half": geo["sym"],
            "R_disk_to_return_x_sigma": st["floating_potentials"][0],
            "current_check": cur, "return_check": back,
            "G_at": G_at, "geo": geo, "x": x, "y": y, "z": z, "s6": s6}


def _pct(samples, axis=0):
    lo, med, hi = np.percentile(samples, [2.5, 50, 97.5], axis=axis)
    return lo, med, hi


def tank_bands(fine, U_num_A, U_num_B, U_num_ref, dlnG_dlna, dlnGref_dlna, unc):
    """Monte Carlo band of G at every recording position, both conditions.

    Drawn independently in every sample: each probe's position along and
    across the recording line and its height; the reference wire's position
    (x, y, height); the electrode diameter (one draw, applied through the
    committed sensitivity d ln G / d ln a); and the grid error (a relative
    normal with SD = GCI / 2, since the GCI is a 95% bound).
    Returns the samples and their 2.5 / 50 / 97.5 percentiles. sigma, the
    temperature and the current are NOT in here: they multiply every reading
    (V = I G / sigma) and are added by transfer_band."""
    rng = np.random.default_rng(int(unc.get("mc_seed", 2026)))
    n = int(unc.get("mc_samples", 4000))
    geo = fine["geo"]
    xy = geo["rec_xy"]; d = geo["direction"]; perp = np.array([-d[1], d[0]])
    m = len(xy)
    sp = unc["probe_position_sd_mm"] * mm; sh = unc["probe_height_sd_mm"] * mm
    da = rng.normal(0, sp, (n, m)); dp = rng.normal(0, sp, (n, m)); dz = rng.normal(0, sh, (n, m))
    P = xy[None, :, :] + da[:, :, None] * d[None, None, :] + dp[:, :, None] * perp[None, None, :]
    # a probe cannot go below the support: its lowest point stays above the disk plane
    dz = np.maximum(dz, -(geo["tip_z"][0] - 0.05 * mm))
    G = fine["G_at"](P.reshape(-1, 2), dz.reshape(-1)).reshape(n, m)
    sr = unc["reference_position_sd_mm"] * mm
    rxy = fine["ref_xy"][None, :] + rng.normal(0, sr, (n, 2))
    rdz = rng.normal(0, sr, n)
    Gref = fine["G_reference"](rxy, fine["ref_z"][0] + rdz, fine["ref_z"][1] + rdz)
    ea = rng.normal(0, unc["electrode_diameter_rel_sd"], n)
    G = G * (1 + ea[:, None] * dlnG_dlna[None, :])
    Gref = Gref * (1 + ea * dlnGref_dlna)
    GA = G * (1 + rng.normal(0, 1, (n, m)) * (U_num_A / 2)[None, :])
    Gref_n = Gref * (1 + rng.normal(0, 1, n) * (U_num_ref / 2))
    GB = (G - Gref[:, None]) * (1 + rng.normal(0, 1, (n, m)) * (U_num_B / 2)[None, :])
    return {"A": GA, "B": GB, "ref": Gref_n, "pct_A": _pct(GA), "pct_B": _pct(GB), "pct_ref": _pct(Gref_n)}


def transfer_band(G_samples, sigma_nominal, unc, seed_offset=1):
    """mV per uA (= kOhm) from G samples: V / I = G / sigma, with the stated
    conductivity-meter and temperature uncertainties drawn once per sample
    (they are common to every reading of a session)."""
    rng = np.random.default_rng(int(unc.get("mc_seed", 2026)) + seed_offset)
    n = G_samples.shape[0]
    sig = sigma_nominal * (1 + rng.normal(0, unc["sigma_meter_rel_sd"], n)) \
        * (1 + unc["sigma_temp_coeff_per_C"] * rng.normal(0, unc["temperature_sd_C"], n))
    shape = (n,) + (1,) * (G_samples.ndim - 1)
    return _pct(G_samples / sig.reshape(shape) / 1e3)


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


def idealized_tank(geo, s6=None):
    """G from the idealized models at the recording positions (tip-averaged).
    The plain formulas put zero potential at infinity. With s6 given, two
    "referenced" versions are added: the same point source read against the
    far reference wire, 1/r - 1/r_ref -- the fairest simple prediction for
    condition B."""
    zt0, zt1 = geo["tip_z"]
    rho = geo["dist"]
    out = {}
    out["point, full space"] = tip_average(lambda xx, yy, zz: point_full_space(np.hypot(rho, zz)),
                                           0, 0, zt0, zt1)
    out["point, half space"] = tip_average(lambda xx, yy, zz: point_half_space(np.hypot(rho, zz)),
                                           0, 0, zt0, zt1)
    out["disk, half space"] = tip_average(lambda xx, yy, zz: disk_half_space(rho, zz, geo["a"]),
                                          0, 0, zt0, zt1)
    if s6 is not None:
        rx, ry = np.asarray(s6["reference_xy_mm"], float) * mm
        rz = np.mean(s6["reference_exposed_z_mm"]) * mm
        r_ref = float(np.sqrt((rx - geo["sx"]) ** 2 + (ry - geo["sy"]) ** 2 + (rz - geo["pt"]) ** 2))
        out["point, full space, referenced"] = out["point, full space"] - point_full_space(r_ref)
        out["point, half space, referenced"] = out["point, half space"] - point_half_space(r_ref)
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
