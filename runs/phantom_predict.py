"""Blind predictions for the saline phantoms -- the project's PRIMARY
experimental validation (research plan v16 Parts 6-7 + the Nov 15 experiment
list). Run this with the MEASURED apparatus, check the output, and COMMIT +
TAG it before any phantom measurement is taken.

    python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1
    python runs/phantom_predict.py --quick               coarse grids, testing only

First: copy notes/phantom_setup_template.json to notes/phantom_setup.json,
replace every ASSUMED value with the measured one, delete any experiment that
will not be built, and (Part 7) run runs/traverse_design.py.

EVERY PREDICTION IS A BAND, in mV per uA (current-normalized) and as the
geometric factor G = phi sigma / I (1/m; volts = I_meas G / sigma_meas).
The band is a Monte Carlo over the measured inputs (probe position and
height, electrode diameter, reference-wire position) with the grid
uncertainty (GCI) drawn in the same samples; conductivity, temperature and
current enter through V = I G / sigma.

PART 6 EXPERIMENTS (each layout under "experiments" in the setup file)
  a. E1_distance     potential against distance (12+ points), both referencing
                     conditions, full model vs the point-source formulas
  b. E2_no_plate     the same electrode with no plate; the committed quantity
                     is the RATIO plate / no plate at each distance (about 2)
  c. E3_return_*     three return positions (far wall = E1, top sheet, ring):
                     the committed quantity is each reading's ratio to the
                     far-wall reading -- 1 near the electrode, far from 1 far
                     away. This is the array's return-path finding at phantom
                     scale. A point source predicts 1 everywhere.
  d. reciprocity     no model input: swapping the driven pair and the measured
                     pair must give the same V / I in ANY layout (checked on
                     the solver in tests/test_phantom_layouts.py)
  e. two salines     G does not depend on sigma, so readings x sigma / I must
                     collapse; the committed number is the repositioning band
  A - B difference   one value per layout, the same at every position

PART 7: the traverse across the agar interface and the gradient ratio.

COVERAGE RULE (fixed here, before any measurement), 95%:
  a reading is covered if it lies between the committed band limits
  I G_lo / sigma and I G_hi / sigma, each widened in quadrature by
  2 sqrt(u_sigma^2 + u_I^2 + u_gain^2) |V_pred| and then by 2 x the reading's
  noise floor. u_sigma is the session's measured conductivity uncertainty
  (meter, drift), u_I the current's, u_gain the scope's. Derived ratios use
  their own committed band, widened by the two readings' noise.
  runs/phantom_compare.py applies exactly this rule, and the pass criteria in
  pass_criteria.json (written here from notes/phantom_pass_criteria.md).
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phantom import (load_setup, solve_tank, solve_sphere, idealized_tank, gci, fd_ratio,
                     experiment_names, layout_setup, tank_bands, transfer_band, _pct, mm)
from phantom_criteria import resolve_criteria, k_min
from scope import lsb_from_settings, traces_needed

args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else \
        ("phantom_quick" if QUICK else "phantom-prediction-v1")
if "--label" in sys.argv:
    args = [a for a in args if a != label]
setup_path = args[0] if args else None
setup = load_setup(setup_path)
unc = dict(setup["uncertainties"])
inst = setup["instrument"]
if QUICK:
    unc["mc_samples"] = 500

OUT = ROOT / "results" / "phantom_predictions" / label
FIG = ROOT / "figures" / "phantom_predictions" / label
if OUT.exists() and any(OUT.iterdir()) and not QUICK:
    sys.exit(f"{OUT.relative_to(ROOT)} already exists -- predictions are never overwritten. "
             f"Use --label to make a new set.")
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
FACTORS = [8.0, 4.0, 2.0] if QUICK else [4.0, 2.0, 1.0]      # coarse -> fine, ratio 2
t0 = time.time()
print(f"setup: {setup_path or 'built-in defaults (ASSUMED geometry)'}; label {label}"
      + ("  [QUICK: not for committing]" if QUICK else ""))
if setup_path is None and not QUICK:
    print("WARNING: no setup file -- these use ASSUMED geometry and must not be tagged.")

I_max = inst["generator_max_amplitude_V"] / inst["series_resistor_ohm"]
lsb = lsb_from_settings(inst["scope_min_volts_per_div"], inst["scope_bits"],
                        inst["scope_full_scale_divisions"])
print(f"instrument (from the setup file): largest current {I_max*1e6:.0f} uA; ADC step at the smallest "
      f"V/div {lsb*1e6:.1f} uV; assumed single-sample noise {inst['scope_single_sample_noise_V']*1e6:.0f} uV")


def traces_for(V, rel=0.02):
    """Traces to average so the noise floor is `rel` of the predicted signal V."""
    n, ok = traces_needed(inst["scope_single_sample_noise_V"], abs(V) * rel,
                          inst["scope_samples_in_plateau"], 1.0, lsb)
    return n if ok else None


# ================================================================== Part 6
names = experiment_names(setup) or [None]
sig = setup["part6"]["sigma_nominal_S_per_m"]
experiments, samples = {}, {}
for name in names:
    s6 = layout_setup(setup, name)
    desc = setup["experiments"][name].get("description", "") if name else "baseline"
    print(f"\nPART 6, {name or 'baseline'}: {desc}")
    runs6 = [solve_tank(setup, f, s6=s6) for f in FACTORS]
    GA = [r["G_condition_A"] for r in runs6]
    GB = [r["G_condition_B"] for r in runs6]
    DIF = [r["G_reference_wire"] for r in runs6]             # A - B, position independent
    UA, pA = gci(GA[2], GA[1], GA[0])
    # condition B can pass through zero far away: its grid uncertainty is kept
    # ABSOLUTE (1/m) and expressed relative to |G_B| with a floor
    eB21, eB32 = GB[1] - GB[2], GB[0] - GB[1]
    UB_rel, pB = gci(GB[2], GB[1], GB[0])
    UB_abs = np.abs(UB_rel * GB[2])
    UD, pD = gci(np.array([DIF[2]]), np.array([DIF[1]]), np.array([DIF[0]]))
    fine = runs6[2]
    geo = fine["geo"]
    # sensitivity to the electrode diameter (middle grid)
    s_hi = json.loads(json.dumps(s6)); s_hi["stim_disk_diameter_mm"] *= 1.2
    s_lo = json.loads(json.dumps(s6)); s_lo["stim_disk_diameter_mm"] /= 1.2
    r_hi = solve_tank(setup, FACTORS[1], verbose=False, s6=s_hi)
    r_lo = solve_tank(setup, FACTORS[1], verbose=False, s6=s_lo)
    dlna = (np.log(r_hi["G"]) - np.log(r_lo["G"])) / (2 * np.log(1.2))
    dlna_ref = float((np.log(abs(r_hi["G_reference_wire"])) - np.log(abs(r_lo["G_reference_wire"])))
                     / (2 * np.log(1.2))) if abs(DIF[2]) > 1e-9 else 0.0
    # the absolute grid error of B, drawn as a relative error on |G_B| (floored)
    UB_for_mc = UB_abs / np.maximum(np.abs(GB[2]), 1e-12)
    mc = tank_bands(fine, UA, UB_for_mc, float(UD[0]), dlna, dlna_ref, unc)
    samples[name] = mc
    RA = transfer_band(mc["A"], sig, unc); RB = transfer_band(mc["B"], sig, unc)
    Rref = transfer_band(mc["ref"], sig, unc)
    # repositioning band: ratio of two independent placements of the same probe
    rng = np.random.default_rng(int(unc["mc_seed"]) + 7)
    perm = rng.permutation(mc["A"].shape[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        repA = _pct(mc["A"] / mc["A"][perm]); repB = _pct(mc["B"] / mc["B"][perm])
    ideal = idealized_tank(geo, s6)
    rows = []
    print(f"{'mm':>5} {'A: mV/uA  [95% band]':>28} {'B: mV/uA  [95% band]':>30} {'grid A/B':>10} "
          f"{'pt full':>8} {'pt half':>8} {'disk':>8}")
    for k, dist in enumerate(geo["dist"] / mm):
        row = {"distance_mm": float(dist),
               "G_A_per_m": float(GA[2][k]), "G_A_lo_per_m": float(mc["pct_A"][0][k]),
               "G_A_hi_per_m": float(mc["pct_A"][2][k]),
               "G_B_per_m": float(GB[2][k]), "G_B_lo_per_m": float(mc["pct_B"][0][k]),
               "G_B_hi_per_m": float(mc["pct_B"][2][k]),
               "U_num_A_rel": float(UA[k]), "order_A": float(pA[k]),
               "U_num_B_abs_per_m": float(UB_abs[k]), "order_B": float(pB[k]),
               "R_A_mV_per_uA": [float(RA[0][k]), float(RA[1][k]), float(RA[2][k])],
               "R_B_mV_per_uA": [float(RB[0][k]), float(RB[1][k]), float(RB[2][k])],
               "dlnG_dln_diameter": float(dlna[k]),
               "reposition_ratio_A_band": [float(repA[0][k]), float(repA[2][k])],
               "reposition_ratio_B_band": [float(repB[0][k]), float(repB[2][k])],
               **{f"G_{m.replace(', ', '_').replace(' ', '_')}_per_m": float(v[k]) for m, v in ideal.items()},
               "G_A_by_grid": [float(g[k]) for g in GA], "G_B_by_grid": [float(g[k]) for g in GB],
               "V_A_at_max_current_V": float(I_max * GA[2][k] / sig),
               "V_B_at_max_current_V": float(I_max * GB[2][k] / sig),
               "traces_for_2pct_noise_B": traces_for(I_max * GB[2][k] / sig)}
        rows.append(row)
        print(f"{dist:5.1f} {RA[1][k]:9.4f} [{RA[0][k]:8.4f},{RA[2][k]:8.4f}] "
              f"{RB[1][k]:10.4f} [{RB[0][k]:8.4f},{RB[2][k]:8.4f}] {UA[k]*100:4.1f}/{UB_abs[k]/sig:5.2f}u "
              f"{ideal['point, full space'][k]/sig/1e3:8.4f} {ideal['point, half space'][k]/sig/1e3:8.4f} "
              f"{ideal['disk, half space'][k]/sig/1e3:8.4f}")
    difference = {"G_A_minus_B_per_m": float(DIF[2]), "G_lo_per_m": float(mc["pct_ref"][0]),
                  "G_hi_per_m": float(mc["pct_ref"][2]), "U_num_rel": float(UD[0]), "order": float(pD[0]),
                  "R_mV_per_uA": [float(Rref[0]), float(Rref[1]), float(Rref[2])],
                  "G_by_grid": list(map(float, DIF)),
                  "note": "V_A - V_B = I G / sigma at EVERY position (position independent)"}
    print(f"  (grid column: A in %, B in uV per uA.) A - B, the same at every distance: "
          f"{Rref[1]:.4f} mV/uA [{Rref[0]:.4f}, {Rref[2]:.4f}]; grid U {UD[0]*100:.1f}%")
    experiments[name or "baseline"] = {
        "description": desc, "setup_part6": s6, "rows": rows, "difference_A_minus_B": difference,
        "R_disk_to_return_x_sigma_by_grid": [r["R_disk_to_return_x_sigma"] for r in runs6],
        "nodes_by_grid": [r["nodes"] for r in runs6],
        "current_check": fine["current_check"], "return_check": fine["return_check"],
        "symmetry_half_tank": bool(fine["symmetry_half"])}
    del runs6, fine, r_hi, r_lo

first = names[0] or "baseline"
dist_mm = [r["distance_mm"] for r in experiments[first]["rows"]]

# ---------------------------------------------------------------- derived ratios
def ratio_band(num, den, G_num, G_den):
    """Band of num / den from independent Monte Carlo samples; 'usable' only
    where both readings are well away from zero."""
    with np.errstate(divide="ignore", invalid="ignore"):
        q = num / den
    lo, med, hi = _pct(q)
    nominal = G_num / np.where(np.abs(G_den) > 0, G_den, np.nan)
    dlo, _, dhi = _pct(den); nlo, _, nhi = _pct(num)
    usable = (dlo > 0) & (nlo > 0) & ((hi - lo) < 0.5 * np.abs(nominal))
    return [{"distance_mm": float(d), "ratio": float(nominal[k]), "lo": float(lo[k]), "hi": float(hi[k]),
             "usable": bool(usable[k])} for k, d in enumerate(dist_mm)]

derived = {}
E = experiments
def GBn(n): return np.array([r["G_B_per_m"] for r in E[n]["rows"]])
def GAn(n): return np.array([r["G_A_per_m"] for r in E[n]["rows"]])
if "E1_distance" in E and "E2_no_plate" in E:
    derived["plate_ratio"] = {
        "definition": "reading with the plate (E1_distance) / reading with no plate (E2_no_plate), "
                      "same distance, each divided by its own current and multiplied by its own sigma",
        "simple_model_value": 2.0,
        "simple_model": "point source: 1/(2 pi sigma r) on an insulating plane vs 1/(4 pi sigma r) in open saline",
        "condition_B": ratio_band(samples["E1_distance"]["B"], samples["E2_no_plate"]["B"],
                                  GBn("E1_distance"), GBn("E2_no_plate")),
        "condition_A": ratio_band(samples["E1_distance"]["A"], samples["E2_no_plate"]["A"],
                                  GAn("E1_distance"), GAn("E2_no_plate"))}
    print("\nPLATE RATIO (E1 / E2, condition B): " + "  ".join(
        f"{q['distance_mm']:g}mm {q['ratio']:.2f}" + ("" if q["usable"] else "*")
        for q in derived["plate_ratio"]["condition_B"]) + "   (* = too close to zero to use)")
rets = [n for n in E if n.startswith("E3_return")]
if "E1_distance" in E and rets:
    derived["return_ratio"] = {
        "definition": "condition-B reading with this return / condition-B reading with the far-wall "
                      "return (E1_distance), same distance",
        "simple_model_value": 1.0,
        "simple_model": "point source (and any model with the return at infinity): no dependence on the return"}
    for n in rets:
        derived["return_ratio"][n] = ratio_band(samples[n]["B"], samples["E1_distance"]["B"],
                                                GBn(n), GBn("E1_distance"))
        print(f"RETURN RATIO ({n} / far wall, condition B): " + "  ".join(
            f"{q['distance_mm']:g}mm {q['ratio']:.2f}" + ("" if q["usable"] else "*")
            for q in derived["return_ratio"][n]))
    allB = np.array([GBn(n) for n in ["E1_distance"] + rets])
    spread = (allB.max(0) - allB.min(0)) / np.abs(allB.mean(0))
    derived["return_spread_rel"] = [{"distance_mm": d, "spread": float(v)} for d, v in zip(dist_mm, spread)]
    print("spread between the return positions (max - min) / mean: " + "  ".join(
        f"{d:g}mm {v*100:.0f}%" for d, v in zip(dist_mm, spread)))
del samples

# ================================================================== Part 7
print("\nPART 7: agar sphere, traverse across the interface")
s7 = setup["part7"]
ratio0 = s7["sigma_in_nominal_S_per_m"] / s7["sigma_out_nominal_S_per_m"]
ideal_R = 1 / ratio0                                       # sigma_out / sigma_in
I7 = max(s7["current_nominal_A"])
part7 = []
for i, off in enumerate(s7["stim_offsets_mm"]):
    runs7 = [solve_sphere(setup, i, f) for f in FACTORS]
    dist = runs7[0]["distances_m"]
    Ra = runs7[0]["interface_m"]
    G7 = [r["G"] for r in runs7]
    U7, p7 = gci(G7[2], G7[1], G7[0])
    R_by_grid = [fd_ratio(dist, g, Ra)[0] for g in G7]
    UR, pR = gci(np.array([R_by_grid[2]]), np.array([R_by_grid[1]]), np.array([R_by_grid[0]]))
    _, idx = fd_ratio(dist, G7[2], Ra)
    fr = runs7[2]
    pos = s7["position_uncertainty_mm"] * mm
    noise = s7["voltage_noise_V"]
    pts = dist[list(idx)]
    sig_out = s7["sigma_out_nominal_S_per_m"]
    # Monte Carlo: every traverse point's position (1 SD = the stated +-),
    # the grid error, and (for the ratio only) the voltage noise
    rng = np.random.default_rng(int(unc["mc_seed"]) + 100 + i)
    n = int(unc["mc_samples"])
    dpos = rng.normal(0, pos, (n, len(dist)))
    Gs = fr["G_traverse"]((dist[None, :] + dpos).reshape(-1)).reshape(n, len(dist))
    Gs = Gs * (1 + rng.normal(0, 1, (n, len(dist))) * (U7 / 2)[None, :])
    lo7, _, hi7 = _pct(Gs)
    def ratio_of(V):
        V = np.atleast_2d(V)
        return ((V[:, 1] - V[:, 0]) / (pts[1] - pts[0])) / ((V[:, 3] - V[:, 2]) / (pts[3] - pts[2]))
    # the ratio uses DIFFERENCES of neighbouring points, where the grid error is
    # almost entirely common; it gets its own GCI (UR) instead of the per-point one
    Gp = fr["G_traverse"]((pts[None, :] + dpos[:, list(idx)]).reshape(-1)).reshape(n, 4)
    Rs_pos = ratio_of(Gp) * (1 + rng.normal(0, 1, n) * UR[0] / 2)
    Vn = I7 * Gp / sig_out + rng.normal(0, noise, (n, 4))
    Rs_all = ratio_of(Vn) * (1 + rng.normal(0, 1, n) * UR[0] / 2)
    R0 = float(R_by_grid[2])
    bp, bn = _pct(Rs_pos), _pct(Rs_all)
    V = I7 * fr["G_traverse"](pts) / sig_out
    # dependence on the conductivity ratio (middle grid)
    hi = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 * 1.2, verbose=False)
    lo = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 / 1.2, verbose=False)
    dlnG = (np.log(hi["G"]) - np.log(lo["G"])) / (2 * np.log(1.2))
    dlnR = (np.log(fd_ratio(dist, hi["G"], Ra)[0]) - np.log(fd_ratio(dist, lo["G"], Ra)[0])) / (2 * np.log(1.2))
    gap = R0 / ideal_R - 1
    # the simple alternative: no interface at all (agar = saline), same points
    uni = solve_sphere(setup, i, FACTORS[1], ratio=1.0, verbose=False)
    R_uniform = float(fd_ratio(dist, uni["G"], Ra)[0])
    U_pos = float(max(R0 - bp[0], bp[2] - R0) / R0)
    U_all = float(max(R0 - bn[0], bn[2] - R0) / R0)
    U_noise = float(np.sqrt(max(U_all ** 2 - U_pos ** 2, 0.0)))
    rec = {"stim_offset_mm": off, "ratio_in_out_nominal": ratio0,
           "traverse_distances_mm": (dist / mm).tolist(), "interface_mm": Ra / mm,
           "inside": (dist < Ra).tolist(),
           "G_per_m": G7[2].tolist(), "G_lo_per_m": lo7.tolist(), "G_hi_per_m": hi7.tolist(),
           "U_num_rel": U7.tolist(),
           "U_pos_rel": (np.maximum(G7[2] - lo7, hi7 - G7[2]) / G7[2]).tolist(),
           "dlnG_dln_ratio": dlnG.tolist(), "G_by_grid": [g.tolist() for g in G7],
           "gradient_ratio": {
               "points_used_mm": (pts / mm).tolist(),
               "fd_prediction": R0, "U_num_rel": float(UR[0]),
               "band_position_and_grid": [float(bp[0]), float(bp[2])],
               "band_with_assumed_noise": [float(bn[0]), float(bn[2])],
               "U_pos_rel": U_pos, "U_noise_rel_at_assumed_noise": U_noise,
               "assumed_noise_V": noise, "current_A": I7,
               "ideal_sigma_out_over_sigma_in": float(ideal_R),
               "uniform_medium_ratio": R_uniform,
               "gap_fd_vs_ideal_rel": float(gap),
               "dlnR_dln_ratio": float(dlnR), "by_grid": list(map(float, R_by_grid))}}
    part7.append(rec)
    print(f"stim at {off} mm: traverse G {G7[2].max():.2f}-{G7[2].min():.2f} 1/m (max U_num "
          f"{np.max(U7)*100:.1f}%)")
    print(f"  GRADIENT RATIO from points {', '.join(f'{v/mm:.1f}' for v in pts)} mm: predicted "
          f"{R0:.4f}, band [{bp[0]:.4f}, {bp[2]:.4f}] (positions + grid); ideal sigma_out/sigma_in "
          f"{ideal_R:.4f}; gap {gap*100:+.1f}%")
    print(f"  with {noise*1e6:.1f} uV noise per reading at {I7*1e6:.0f} uA: band [{bn[0]:.4f}, {bn[2]:.4f}] "
          f"(differences measured: {abs(V[1]-V[0])*1e6:.1f} uV inside, {abs(V[3]-V[2])*1e6:.1f} uV outside)")

# ================================================================== criteria, save
criteria = resolve_criteria(experiments, derived, part7, setup,
                            (ROOT / "notes" / "phantom_pass_criteria.md"))
print("\nPASS CRITERIA resolved for this geometry (rules: notes/phantom_pass_criteria.md):")
for line in criteria["summary_lines"]:
    print("  " + line)

summary = {
    "label": label, "quick": QUICK, "created": time.strftime("%Y-%m-%d %H:%M"),
    "plan_version": "v16 + Nov 15 experiments", "setup": setup, "setup_file": setup_path,
    "grid_factors": FACTORS, "mc_samples": int(unc["mc_samples"]),
    "coverage_rule": ("Covered if the reading lies between I G_lo / sigma and I G_hi / sigma (committed "
                      "Monte Carlo band: probe position and height, electrode diameter, reference "
                      "position, grid uncertainty), each half-width widened in quadrature by "
                      "2 sqrt(u_sigma^2 + u_I^2 + u_gain^2) |V_pred|, then by 2 x the noise floor. "
                      "Part 7: G and the ratio corrected to the measured sigma_in/sigma_out with the "
                      "committed d ln / d ln(ratio); the ratio band adds the session's noise."),
    "instrument_used": {"max_current_A": I_max, "adc_step_V": lsb},
    "experiments": experiments, "part6": experiments[first], "derived": derived,
    "part7": part7,
}
(OUT / "predictions.json").write_text(json.dumps(summary, indent=2))
(OUT / "pass_criteria.json").write_text(json.dumps(criteria, indent=2))
for name, ex in experiments.items():
    hdr = ["distance_mm", "R_A_mV_per_uA", "R_A_lo", "R_A_hi", "R_B_mV_per_uA", "R_B_lo", "R_B_hi",
           "G_A_per_m", "G_A_lo_per_m", "G_A_hi_per_m", "G_B_per_m", "G_B_lo_per_m", "G_B_hi_per_m",
           "G_point_full_space_per_m", "G_point_half_space_per_m", "G_disk_half_space_per_m"]
    lines = [",".join(hdr)]
    for r in ex["rows"]:
        v = [r["distance_mm"], r["R_A_mV_per_uA"][1], r["R_A_mV_per_uA"][0], r["R_A_mV_per_uA"][2],
             r["R_B_mV_per_uA"][1], r["R_B_mV_per_uA"][0], r["R_B_mV_per_uA"][2]] + [r[k] for k in hdr[7:]]
        lines.append(",".join(f"{x:.6g}" for x in v))
    D = ex["difference_A_minus_B"]
    lines.append(f"# mV/uA columns are at the nominal {sig} S/m with the stated input uncertainties")
    lines.append(f"# difference A-B (all positions): G {D['G_A_minus_B_per_m']:.6g} 1/m "
                 f"[{D['G_lo_per_m']:.6g}, {D['G_hi_per_m']:.6g}]")
    (OUT / f"part6_{name}_predictions.csv").write_text("\n".join(lines) + "\n")
for k, rec in enumerate(part7):
    L = ["distance_from_agar_centre_mm,inside_agar,G_per_m,G_lo_per_m,G_hi_per_m,U_num_rel,dlnG_dln_ratio"]
    L += [f"{a:.4g},{int(b)},{c:.6g},{l_:.6g},{h_:.6g},{dd:.4g},{f_:.4g}" for a, b, c, l_, h_, dd, f_ in
          zip(rec["traverse_distances_mm"], rec["inside"], rec["G_per_m"], rec["G_lo_per_m"],
              rec["G_hi_per_m"], rec["U_num_rel"], rec["dlnG_dln_ratio"])]
    gr = rec["gradient_ratio"]
    L.append(f"# gradient ratio: predicted {gr['fd_prediction']:.6g}, band "
             f"[{gr['band_position_and_grid'][0]:.6g}, {gr['band_position_and_grid'][1]:.6g}]; ideal "
             f"{gr['ideal_sigma_out_over_sigma_in']:.6g}; gap {gr['gap_fd_vs_ideal_rel']:.4g}; "
             f"dlnR/dln(ratio) {gr['dlnR_dln_ratio']:.4g}")
    (OUT / f"part7_stim{k}_predictions.csv").write_text("\n".join(L) + "\n")

# ================================================================== figure
COL = {"E1_distance": "#2a78d6", "E2_no_plate": "#eb6834", "E3_return_top": "#1baf7a",
       "E3_return_ring": "#7b4fd0", "baseline": "#2a78d6"}
fig, axs = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
ax = axs[0, 0]
ex = experiments[first]
dd = np.array(dist_mm)
for key, lab, c in (("R_A_mV_per_uA", "A: ground at the return", "#2a78d6"),
                    ("R_B_mV_per_uA", "B: far reference wire", "#7b4fd0")):
    b = np.array([r[key] for r in ex["rows"]])
    ok = b[:, 0] > 0
    ax.fill_between(dd[ok], b[ok, 0], b[ok, 2], color=c, alpha=0.25, lw=0)
    ax.loglog(dd[ok], b[ok, 1], "o-", color=c, lw=1.8, ms=3.5, label=lab)
for m, c, ls in (("point_full_space", "#eb6834", "-"), ("point_half_space", "#1baf7a", "-"),
                 ("disk_half_space", "0.4", ":")):
    ax.loglog(dd, np.array([r[f"G_{m}_per_m"] for r in ex["rows"]]) / sig / 1e3, ls, color=c, lw=1.3,
              label=m.replace("_", " "))
ax.set_xlabel("distance from the stimulating disk (mm)"); ax.set_ylabel("potential per current (mV per uA)")
ax.set_title(f"a. {first}: full model (95% bands) vs point source", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=7.5)
ax = axs[0, 1]
if "plate_ratio" in derived:
    q = [v for v in derived["plate_ratio"]["condition_B"] if v["usable"]]
    x_ = [v["distance_mm"] for v in q]
    ax.fill_between(x_, [v["lo"] for v in q], [v["hi"] for v in q], color="#2a78d6", alpha=0.25, lw=0)
    ax.semilogx(x_, [v["ratio"] for v in q], "o-", color="#2a78d6", lw=1.8, ms=3.5, label="full model, 95% band")
    ax.axhline(2.0, color="#eb6834", lw=1.3, label="point source: exactly 2")
    ax.set_ylabel("reading with plate / reading with no plate")
    ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("distance (mm)"); ax.set_title("b. the insulating plate (condition B)", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5)
ax = axs[1, 0]
if "return_ratio" in derived:
    for n in rets:
        q = [v for v in derived["return_ratio"][n] if v["usable"]]
        x_ = [v["distance_mm"] for v in q]
        ax.fill_between(x_, [v["lo"] for v in q], [v["hi"] for v in q], color=COL.get(n, "0.3"), alpha=0.25, lw=0)
        ax.semilogx(x_, [v["ratio"] for v in q], "o-", color=COL.get(n, "0.3"), lw=1.8, ms=3.5,
                    label=n.replace("E3_return_", "") + " / far wall")
    ax.axhline(1.0, color="#eb6834", lw=1.3, label="point source: exactly 1")
    ax.set_ylabel("reading / reading with far-wall return"); ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("distance (mm)"); ax.set_title("c. where the return electrode is (condition B)", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5)
ax = axs[1, 1]
for rec, c in zip(part7, ["#2a78d6", "#eb6834"]):
    g = np.array(rec["G_per_m"]); x_ = np.array(rec["traverse_distances_mm"])
    ax.fill_between(x_, np.array(rec["G_lo_per_m"]) - g[-1], np.array(rec["G_hi_per_m"]) - g[-1],
                    color=c, alpha=0.25, lw=0)
    ax.plot(x_, g - g[-1], "o-", color=c, lw=1.8, ms=3.5,
            label=f"stim at {rec['stim_offset_mm'][0]:.0f} mm: ratio {rec['gradient_ratio']['fd_prediction']:.3f}")
ax.axvline(part7[0]["interface_mm"], color="0.5", ls="--", lw=1)
ax.set_xlabel("distance from the agar centre along the traverse (mm)")
ax.set_ylabel("G - G(last point) (1/m)")
ax.set_title(f"Part 7: traverse; ideal gradient ratio = {ideal_R:.3f}", fontsize=10)
ax.grid(True, color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"Phantom blind predictions [{label}]" + (" -- QUICK, NOT FOR COMMITTING" if QUICK else "")
             + ("" if setup_path else " -- ASSUMED GEOMETRY"))
fig.savefig(FIG / "phantom_predictions.png", dpi=200)
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
if not QUICK and setup_path:
    print(f"NEXT, BEFORE ANY MEASUREMENT: git add notes/phantom_setup.json notes/phantom_pass_criteria.md "
          f"{OUT.relative_to(ROOT)} {FIG.relative_to(ROOT)} && git commit -m 'Phantom blind prediction' "
          f"&& git tag {label} && git push && git push --tags")
