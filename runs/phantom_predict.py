"""Blind predictions for the saline phantoms (research plan v16, steps 53-54
and 65-68). Run this with the MEASURED apparatus, check the output, and
COMMIT + TAG it before any phantom measurement is taken.

    python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1
    python runs/phantom_predict.py --quick               coarse grids, testing only

First: copy notes/phantom_setup_template.json to notes/phantom_setup.json and
replace every ASSUMED value with the measured one; run
runs/traverse_design.py to choose the traverse points nearest the interface
(step 66) and put them in the setup file.

What is committed (G = phi sigma / I, in 1/m; volts = I_meas G / sigma_meas)

PART 6, every recording distance:
  condition A (ground at return)   G_A, numerical band U_num_A
  condition B (far reference wire) G_B, numerical band U_num_B
  U_pos  change for the stated recording-position and tip-height uncertainty
  DIFFERENCE A - B: one value for the whole session (it does not depend on
  the recording position), with its numerical band and the band from the
  reference wire's position uncertainty. This is the phantom's measurement of
  the return-path term Aim 1 quantifies in the array.
  comparison models (point full space, point half space, disk half space):
  they assume the return is at infinity, so they are compared with B.

PART 7, each stimulating position:
  G along the traverse (condition A), U_num, U_pos
  fd_ratio  the gradient ratio across the interface from the four traverse
            points nearest it -- the same points and differences the
            measurement will use (step 65), with U_num, and its measurement
            uncertainty from +-0.1 mm positions (U_pos) and voltage noise (U_noise)
  ideal ratio sigma_out / sigma_in, and the GAP fd_ratio / ideal - 1, reported
            separately: how much of a disagreement is finite spacing
  d ln(fd_ratio) / d ln(sigma_in / sigma_out) and d ln G / d ln(...): the
            fixed rule that corrects the prediction to the MEASURED
            conductivities (steps 62, 64), so the test has no fitted parameter

COVERAGE RULE (fixed here, before any measurement), 95%:
  relative band U = sqrt( U_num^2 + U_pos^2 + (2 u_sigma)^2 + (2 u_I)^2 )
  with u_sigma, u_I the measured relative standard uncertainties (session
  drift included in u_sigma). Covered if |V_meas - V_pred| <= U V_pred +
  2 x session noise floor. For the gradient ratio: |R_meas - R_pred| <=
  sqrt(U_num^2 + U_pos^2 + U_noise^2 + (2 u_ratio dlnR)^2) R_pred, U_noise
  recomputed from the session's measured noise. runs/phantom_compare.py
  applies exactly this rule.
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
                     design_spacing, mm)

args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else \
        ("phantom_quick" if QUICK else "phantom-prediction-v1")
if "--label" in sys.argv:
    args = [a for a in args if a != label]
setup_path = args[0] if args else None
setup = load_setup(setup_path)

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

# ================================================================== Part 6
print("\nPART 6: uniform tank, two referencing conditions")
s6 = setup["part6"]
runs6 = [solve_tank(setup, f) for f in FACTORS]
GA = [r["G_condition_A"] for r in runs6]
GB = [r["G_condition_B"] for r in runs6]
DIF = [r["G_reference_wire"] for r in runs6]                 # A - B, position independent
UA, pA = gci(GA[2], GA[1], GA[0])
UB, pB = gci(GB[2], GB[1], GB[0])
UD, pD = gci(np.array([DIF[2]]), np.array([DIF[1]]), np.array([DIF[0]]))
fine = runs6[2]
geo = fine["geo"]
dpos = s6["position_uncertainty_mm"] * mm
dz = s6["tip_height_uncertainty_mm"] * mm
d = geo["direction"]
xy = geo["rec_xy"]
Gp = fine["G_at"](xy + dpos * d); Gm = fine["G_at"](np.maximum(xy - dpos * d, 0))
Gu = fine["G_at"](xy, +dz); Gd = fine["G_at"](xy, -dz)
dG_pos = np.sqrt(((Gp - Gm) / 2) ** 2 + ((Gu - Gd) / 2) ** 2)    # absolute, same for A and B
UposA = dG_pos / GA[2]; UposB = dG_pos / GB[2]
# reference-wire position: moves the B curve and the difference by the same amount
ref_xy = np.asarray(s6["reference_xy_mm"], float) * mm
rz0, rz1 = np.asarray(s6["reference_exposed_z_mm"], float) * mm
dref = s6.get("reference_position_uncertainty_mm", 1.0) * mm
shifts = [fine["G_reference"](ref_xy + np.array(v) * dref, rz0, rz1) - DIF[2]
          for v in ([1, 0], [-1, 0], [0, 1], [0, -1])]
shifts.append(fine["G_reference"](ref_xy, rz0 + dref, rz1 + dref) - DIF[2])
shifts.append(fine["G_reference"](ref_xy, rz0 - dref, rz1 - dref) - DIF[2])
dG_ref = float(np.sqrt((max(abs(shifts[0]), abs(shifts[1]))) ** 2 + max(abs(shifts[2]), abs(shifts[3])) ** 2
                       + max(abs(shifts[4]), abs(shifts[5])) ** 2))
ideal = idealized_tank(geo)
sig, Inom = s6["sigma_nominal_S_per_m"], s6["current_nominal_A"]

print(f"{'mm':>5} {'G_A':>8} {'U_num':>6} {'G_B':>8} {'U_num':>6} {'U_pos A/B':>11} "
      f"{'pt full':>8} {'pt half':>8} {'disk':>8}   V_A, V_B at {Inom[0]*1e6:.0f} uA, {sig} S/m")
rows6 = []
for k, dist in enumerate(geo["dist"] / mm):
    row = {"distance_mm": float(dist),
           "G_A_per_m": float(GA[2][k]), "U_num_A_rel": float(UA[k]), "order_A": float(pA[k]),
           "G_B_per_m": float(GB[2][k]), "U_num_B_rel": float(UB[k]), "order_B": float(pB[k]),
           "U_pos_A_rel": float(UposA[k]), "U_pos_B_rel": float(UposB[k]),
           "U_refpos_B_rel": float(dG_ref / GB[2][k]),
           **{f"G_{m.replace(', ', '_').replace(' ', '_')}_per_m": float(v[k]) for m, v in ideal.items()},
           "G_A_by_grid": [float(g[k]) for g in GA], "G_B_by_grid": [float(g[k]) for g in GB]}
    rows6.append(row)
    print(f"{dist:5.1f} {GA[2][k]:8.2f} {UA[k]*100:5.1f}% {GB[2][k]:8.2f} {UB[k]*100:5.1f}% "
          f"{UposA[k]*100:4.1f}/{UposB[k]*100:4.1f}% {ideal['point, full space'][k]:8.2f} "
          f"{ideal['point, half space'][k]:8.2f} {ideal['disk, half space'][k]:8.2f}   "
          f"{Inom[0]*GA[2][k]/sig*1e3:.3f}, {Inom[0]*GB[2][k]/sig*1e3:.3f} mV")
difference = {"G_A_minus_B_per_m": float(DIF[2]), "U_num_rel": float(UD[0]), "order": float(pD[0]),
              "U_refpos_rel": float(dG_ref / DIF[2]), "G_by_grid": list(map(float, DIF)),
              "note": "V_A - V_B = I G / sigma at EVERY position (position independent)"}
print(f"PREDICTED DIFFERENCE A - B: G = {DIF[2]:.2f} 1/m (grid U {UD[0]*100:.1f}%, reference position "
      f"U {dG_ref/DIF[2]*100:.1f}%) -> {Inom[0]*DIF[2]/sig*1e3:.3f} mV at {Inom[0]*1e6:.0f} uA; "
      f"the same at every distance")
print(f"share of the condition-A reading at {geo['dist'][-1]/mm:.0f} mm that is the difference: "
      f"{DIF[2]/GA[2][-1]*100:.0f}%")
R_sig = [r["R_disk_to_return_x_sigma"] for r in runs6]

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
    # measurement uncertainty of the ratio at these exact points
    pos = s7["position_uncertainty_mm"] * mm
    noise = s7["voltage_noise_V"]
    pts = dist[list(idx)]
    V = I7 * fr["G_traverse"](pts) / s7["sigma_out_nominal_S_per_m"]
    slope = I7 * (fr["G_traverse"](pts + 1e-5) - fr["G_traverse"](pts - 1e-5)) / 2e-5 / s7["sigma_out_nominal_S_per_m"]
    def ratio_of(Vv):
        return ((Vv[1] - Vv[0]) / (pts[1] - pts[0])) / ((Vv[3] - Vv[2]) / (pts[3] - pts[2]))
    R0 = ratio_of(V)
    U_pos = float(np.sqrt(sum((ratio_of(V + np.eye(4)[k] * slope[k] * pos) / R0 - 1) ** 2 for k in range(4))))
    U_noise = float(np.sqrt(sum((ratio_of(V + np.eye(4)[k] * noise) / R0 - 1) ** 2 for k in range(4))))
    Upos_G = np.abs(fr["G_traverse"](dist + pos) - fr["G_traverse"](dist - pos)) / 2 / G7[2]
    # dependence on the conductivity ratio (middle grid)
    hi = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 * 1.2, verbose=False)
    lo = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 / 1.2, verbose=False)
    dlnG = (np.log(hi["G"]) - np.log(lo["G"])) / (2 * np.log(1.2))
    dlnR = (np.log(fd_ratio(dist, hi["G"], Ra)[0]) - np.log(fd_ratio(dist, lo["G"], Ra)[0])) / (2 * np.log(1.2))
    gap = R_by_grid[2] / ideal_R - 1
    rec = {"stim_offset_mm": off, "ratio_in_out_nominal": ratio0,
           "traverse_distances_mm": (dist / mm).tolist(), "interface_mm": Ra / mm,
           "inside": (dist < Ra).tolist(),
           "G_per_m": G7[2].tolist(), "U_num_rel": U7.tolist(), "U_pos_rel": Upos_G.tolist(),
           "dlnG_dln_ratio": dlnG.tolist(), "G_by_grid": [g.tolist() for g in G7],
           "gradient_ratio": {
               "points_used_mm": (pts / mm).tolist(),
               "fd_prediction": float(R_by_grid[2]), "U_num_rel": float(UR[0]),
               "U_pos_rel": U_pos, "U_noise_rel_at_assumed_noise": U_noise,
               "assumed_noise_V": noise, "current_A": I7,
               "ideal_sigma_out_over_sigma_in": float(ideal_R),
               "gap_fd_vs_ideal_rel": float(gap),
               "dlnR_dln_ratio": float(dlnR), "by_grid": list(map(float, R_by_grid))}}
    part7.append(rec)
    print(f"stim at {off} mm: traverse G {G7[2].max():.2f}-{G7[2].min():.2f} 1/m (max U_num "
          f"{np.max(U7)*100:.1f}%)")
    print(f"  GRADIENT RATIO from points {', '.join(f'{v/mm:.1f}' for v in pts)} mm: predicted "
          f"{R_by_grid[2]:.4f} (grid U {UR[0]*100:.1f}%); ideal sigma_out/sigma_in {ideal_R:.4f}; "
          f"gap {gap*100:+.1f}%")
    print(f"  measurement uncertainty of the ratio: position {U_pos*100:.1f}%, noise "
          f"{U_noise*100:.1f}% at {noise*1e6:.1f} uV and {I7*1e6:.0f} uA "
          f"(differences measured: {abs(V[1]-V[0])*1e6:.1f} uV inside, {abs(V[3]-V[2])*1e6:.1f} uV outside)")

# ================================================================== save
summary = {
    "label": label, "quick": QUICK, "created": time.strftime("%Y-%m-%d %H:%M"),
    "plan_version": "v16", "setup": setup, "setup_file": setup_path, "grid_factors": FACTORS,
    "coverage_rule": ("Relative 95% band U = sqrt(U_num^2 + U_pos^2 + (2 u_sigma)^2 + (2 u_I)^2); "
                      "covered if |V_meas - V_pred| <= U V_pred + 2 noise_floor. Condition B adds "
                      "U_refpos. Part 7: G and the ratio corrected to the measured sigma_in/sigma_out "
                      "with the committed d ln / d ln(ratio); ratio band sqrt(U_num^2 + U_pos^2 + "
                      "U_noise^2 + (2 u_ratio dlnR)^2) with U_noise from the session noise."),
    "part6": {"rows": rows6, "difference_A_minus_B": difference,
              "R_disk_to_return_x_sigma_by_grid": R_sig,
              "symmetry_half_tank": bool(fine["symmetry_half"])},
    "part7": part7,
}
(OUT / "predictions.json").write_text(json.dumps(summary, indent=2))
hdr = ["distance_mm", "G_A_per_m", "U_num_A_rel", "G_B_per_m", "U_num_B_rel", "U_pos_A_rel",
       "U_pos_B_rel", "U_refpos_B_rel", "G_point_full_space_per_m", "G_point_half_space_per_m",
       "G_disk_half_space_per_m"]
lines = [",".join(hdr)] + [",".join(f"{r[k]:.6g}" for k in hdr) for r in rows6]
lines.append(f"# difference A-B (all positions): G {DIF[2]:.6g} 1/m, U_num {UD[0]:.4g}, "
             f"U_refpos {dG_ref/DIF[2]:.4g}")
(OUT / "part6_predictions.csv").write_text("\n".join(lines) + "\n")
for k, rec in enumerate(part7):
    L = ["distance_from_agar_centre_mm,inside_agar,G_per_m,U_num_rel,U_pos_rel,dlnG_dln_ratio"]
    L += [f"{a:.4g},{int(b)},{c:.6g},{dd:.4g},{e:.4g},{f_:.4g}" for a, b, c, dd, e, f_ in
          zip(rec["traverse_distances_mm"], rec["inside"], rec["G_per_m"], rec["U_num_rel"],
              rec["U_pos_rel"], rec["dlnG_dln_ratio"])]
    gr = rec["gradient_ratio"]
    L.append(f"# gradient ratio: predicted {gr['fd_prediction']:.6g} (U_num {gr['U_num_rel']:.3g}, "
             f"U_pos {gr['U_pos_rel']:.3g}); ideal {gr['ideal_sigma_out_over_sigma_in']:.6g}; "
             f"gap {gr['gap_fd_vs_ideal_rel']:.4g}; dlnR/dln(ratio) {gr['dlnR_dln_ratio']:.4g}")
    (OUT / f"part7_stim{k}_predictions.csv").write_text("\n".join(L) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5), layout="constrained")
ax = axs[0]
dd = geo["dist"] / mm
for G_, U_, lab, c in [(GA[2], np.sqrt(UA ** 2 + UposA ** 2), "A: ground at return", "#2a78d6"),
                       (GB[2], np.sqrt(UB ** 2 + UposB ** 2), "B: far reference wire", "#7b4fd0")]:
    ax.fill_between(dd, G_ * (1 - U_), G_ * (1 + U_), color=c, alpha=0.18, lw=0)
    ax.loglog(dd, G_, "o-", color=c, lw=2, label=lab)
ax.loglog(dd, np.full(len(dd), DIF[2]), "k--", lw=1.2, label="A - B (same at every distance)")
for (m, v), c, ls in zip(ideal.items(), ["#eb6834", "#1baf7a", "0.4"], ["-", "-", ":"]):
    ax.loglog(dd, v, ls, color=c, lw=1.4, label=m)
ax.set_xlabel("distance from the stimulating disk (mm)"); ax.set_ylabel("G = phi sigma / I (1/m)")
ax.set_title("Part 6: both referencing conditions (bands: numerical + position, 95%)", fontsize=9.5)
ax.grid(True, which="both", color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
for rec, c in zip(part7, ["#2a78d6", "#eb6834"]):
    g = np.array(rec["G_per_m"]); x_ = np.array(rec["traverse_distances_mm"])
    ax.plot(x_, g - g[-1], "o-", color=c, lw=2,
            label=f"stim at {rec['stim_offset_mm'][0]:.0f} mm: ratio {rec['gradient_ratio']['fd_prediction']:.3f}")
ax.axvline(part7[0]["interface_mm"], color="0.5", ls="--", lw=1)
ax.set_xlabel("distance from the agar centre along the traverse (mm)")
ax.set_ylabel("G - G(last point) (1/m)")
ax.set_title(f"Part 7: traverse; ideal gradient ratio sigma_out/sigma_in = {ideal_R:.3f}", fontsize=9.5)
ax.grid(True, color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"Phantom blind predictions [{label}]" + (" -- QUICK, NOT FOR COMMITTING" if QUICK else "")
             + ("" if setup_path else " -- ASSUMED GEOMETRY"))
fig.savefig(FIG / "phantom_predictions.png", dpi=200)
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
if not QUICK and setup_path:
    print(f"NEXT, BEFORE ANY MEASUREMENT: git add notes/phantom_setup.json {OUT.relative_to(ROOT)} "
          f"{FIG.relative_to(ROOT)} && git commit -m 'Phantom blind prediction' && git tag {label} "
          f"&& git push && git push --tags")
