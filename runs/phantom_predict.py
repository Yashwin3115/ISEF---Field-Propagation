"""Blind predictions for the saline phantoms (research plan Parts 6 and 7,
steps 52 and 64). Run this, check the output, and COMMIT + TAG it before any
phantom measurement is taken.

    python runs/phantom_predict.py                       defaults (plan v13 + ASSUMED values)
    python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1
    python runs/phantom_predict.py --quick               coarse grids, for testing only

Edit a copy of notes/phantom_setup_template.json to match the apparatus
first (tank, fill depth, plate, return wire, recording line, tip height).

What is committed, for every recording position:
  G = phi * sigma / I  (1/m) from the full model on the finest of three grids
  U_num  Grid Convergence Index of G (three grids, ratio 2, Fs = 1.25)
  U_pos  change in G for the stated position and tip-height uncertainty
  the comparison models (point full space, point half space, disk half space)
Part 7 also: d ln G / d ln(sigma_in/sigma_out), for correcting to the measured
conductivity ratio by the fixed linear rule below.

COVERAGE RULE (fixed here, before any measurement):
  V_pred = I_meas * G / sigma_meas  (Part 7: G corrected to the measured ratio
  with the committed derivative). Relative band, 95%:
    U = sqrt( U_num^2 + U_pos^2 + U_sigma^2 + U_I^2 )
  with U_sigma, U_I the measured relative uncertainties of conductivity and
  current (2 x standard deviation; session drift included in U_sigma).
  A measurement is COVERED if |V_meas - V_pred| <= U * V_pred
  (plus 2 x the session noise floor in volts, added linearly). The headline
  statistic is the covered fraction over all positions, pulse widths and
  amplitudes (analysis/phantom_compare.py applies exactly this rule).
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
from phantom import (load_setup, solve_tank, solve_sphere, idealized_tank, gci, mm,
                     tank_geometry)

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
FACTORS = [8.0, 4.0, 2.0] if QUICK else [4.0, 2.0, 1.0]      # coarse -> fine
t0 = time.time()
print(f"setup: {setup_path or 'built-in defaults (plan v13 + ASSUMED values)'}; label {label}"
      + ("  [QUICK: not for committing]" if QUICK else ""))

# ------------------------------------------------------------------ Part 6
print("\nPART 6: uniform tank")
s6 = setup["part6"]
runs6 = [solve_tank(setup, f) for f in FACTORS]
G3 = [r["G"] for r in runs6]
Ga3 = [r["G_alt_reference"] for r in runs6]
U_num, p_obs = gci(G3[2], G3[1], G3[0])
U_num_a, _ = gci(Ga3[2], Ga3[1], Ga3[0])
fine = runs6[2]
geo = fine["geo"]
dpos = s6["position_uncertainty_mm"] * mm
dz = s6["tip_height_uncertainty_mm"] * mm
d = geo["direction"]
xy = geo["rec_xy"]
Gp = fine["G_at"](xy + dpos * d); Gm = fine["G_at"](np.maximum(xy - dpos * d, 0))
Gu = fine["G_at"](xy, +dz); Gd = fine["G_at"](xy, -dz)
G = G3[2]
U_pos = np.sqrt(((Gp - Gm) / 2) ** 2 + ((Gu - Gd) / 2) ** 2) / G
ideal = idealized_tank(geo)
sig, Inom = s6["sigma_nominal_S_per_m"], s6["current_nominal_A"]

print(f"{'dist mm':>7} {'G full':>8} {'U_num':>6} {'U_pos':>6} {'G alt ref':>9} "
      f"{'pt full':>8} {'pt half':>8} {'disk half':>9}   V at {Inom[0]*1e6:.0f} uA, {sig} S/m")
rows6 = []
for k, dist in enumerate(geo["dist"] / mm):
    row = {"distance_mm": float(dist), "G_full_per_m": float(G[k]),
           "U_num_rel": float(U_num[k]), "observed_order": float(p_obs[k]),
           "U_pos_rel": float(U_pos[k]),
           "G_full_alt_reference_per_m": float(Ga3[2][k]), "U_num_alt_rel": float(U_num_a[k]),
           **{f"G_{m.replace(', ', '_').replace(' ', '_')}_per_m": float(v[k]) for m, v in ideal.items()},
           "G_by_grid": [float(g[k]) for g in G3]}
    rows6.append(row)
    print(f"{dist:7.1f} {G[k]:8.2f} {U_num[k]*100:5.1f}% {U_pos[k]*100:5.1f}% {Ga3[2][k]:9.2f} "
          f"{ideal['point, full space'][k]:8.2f} {ideal['point, half space'][k]:8.2f} "
          f"{ideal['disk, half space'][k]:9.2f}   {Inom[0]*G[k]/sig*1e3:.3f} mV")
R_sig = [r["R_disk_to_return_x_sigma"] for r in runs6]
print(f"disk-to-return resistance x sigma: {R_sig[-1]:.1f} 1/m "
      f"(grids {', '.join(f'{v:.1f}' for v in R_sig)}) -> {R_sig[-1]/sig/1e3:.3f} kOhm at {sig} S/m")
print(f"share of the reading at {geo['dist'][-1]/mm:.0f} mm that is the return-wire offset "
      f"(full - disk half space): {(G[-1]-ideal['disk, half space'][-1])/G[-1]*100:.0f}%")

# ------------------------------------------------------------------ Part 7
print("\nPART 7: agar sphere")
s7 = setup["part7"]
ratio0 = s7["sigma_in_nominal_S_per_m"] / s7["sigma_out_nominal_S_per_m"]
part7 = []
for i, off in enumerate(s7["stim_offsets_mm"]):
    runs7 = [solve_sphere(setup, i, f) for f in FACTORS]
    G7 = [r["G"] for r in runs7]
    U7, _ = gci(G7[2], G7[1], G7[0])
    D7 = [g - g.mean() for g in G7]                    # offset-free differences
    UD7 = np.abs(D7[2] - D7[1]) * 1.25 / 3 / np.maximum(np.abs(D7[2]), 1e-12)   # p=2 -> r^p-1=3
    L7 = [r["G_line"] for r in runs7]
    UL7, _ = gci(L7[2], L7[1], L7[0]) if len(L7[2]) else (np.array([]), None)
    # ratio sensitivity on the middle grid
    hi = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 * 1.2, verbose=False)
    lo = solve_sphere(setup, i, FACTORS[1], ratio=ratio0 / 1.2, verbose=False)
    dlnG = (np.log(hi["G"]) - np.log(lo["G"])) / (2 * np.log(1.2))
    dlnGl = (np.log(hi["G_line"]) - np.log(lo["G_line"])) / (2 * np.log(1.2)) if len(L7[2]) else []
    dD = (hi["G"] - hi["G"].mean() - (lo["G"] - lo["G"].mean())) / (2 * np.log(1.2))
    # position uncertainty: radial shift of each tip
    fr = runs7[2]
    P = fr["positions_m"]; u = P / np.linalg.norm(P, axis=1, keepdims=True)
    dp = s6["position_uncertainty_mm"] * mm * 5        # ASSUMED: +-0.5 mm on a curved surface
    Upos7 = np.abs(fr["G_points"](P + dp * u, u) - fr["G_points"](P - dp * u, u)) / 2 / G7[2]
    part7.append({"stim_offset_mm": off, "ratio_nominal": ratio0,
                  "positions_mm": (P / mm).tolist(),
                  "G_per_m": G7[2].tolist(), "U_num_rel": U7.tolist(), "U_pos_rel": Upos7.tolist(),
                  "dlnG_dln_ratio": dlnG.tolist(),
                  "G_minus_mean_per_m": D7[2].tolist(), "U_num_minus_mean_abs":
                      (UD7 * np.abs(D7[2])).tolist(), "d_G_minus_mean_dln_ratio": dD.tolist(),
                  "line_x_mm": (fr["line_x_m"] / mm).tolist(), "G_line_per_m": L7[2].tolist(),
                  "U_num_line_rel": list(map(float, UL7)), "dlnG_line_dln_ratio": list(map(float, dlnGl)),
                  "G_by_grid": [g.tolist() for g in G7]})
    print(f"stim offset {off} mm: G over the 12 surface positions {G7[2].min():.2f}-{G7[2].max():.2f} 1/m "
          f"(U_num {np.max(U7)*100:.1f}%); variation between positions "
          f"+-{np.ptp(G7[2])/2/np.mean(G7[2])*100:.1f}% of the reading")
    if len(L7[2]):
        print(f"  line across the agar boundary (x = {s7['line_positions_mm'][0]}-"
              f"{s7['line_positions_mm'][-1]} mm): G {L7[2][0]:.1f} -> {L7[2][-1]:.1f} 1/m")

# ------------------------------------------------------------------ save
summary = {
    "label": label, "quick": QUICK, "created_s": time.time(),
    "setup": setup, "setup_file": setup_path, "grid_factors": FACTORS,
    "coverage_rule": ("V_pred = I_meas * G / sigma_meas (Part 7: G * (ratio_meas/ratio_nominal)"
                      "^dlnG_dln_ratio). Covered if |V_meas - V_pred| <= sqrt(U_num^2 + U_pos^2 + "
                      "U_sigma^2 + U_I^2) * V_pred + 2 * noise_floor_V; U_sigma, U_I = 2 x measured "
                      "relative standard deviations (session drift included in U_sigma)."),
    "part6": {"rows": rows6, "R_disk_to_return_x_sigma_by_grid": R_sig,
              "alt_reference_point_G_by_grid": [r["G_at_alt_reference_point"] for r in runs6],
              "symmetry_half_tank": bool(fine["symmetry_half"])},
    "part7": part7,
}
(OUT / "predictions.json").write_text(json.dumps(summary, indent=2))
hdr = "distance_mm,G_full_per_m,U_num_rel,U_pos_rel,G_full_alt_reference_per_m," \
      "G_point_full_space_per_m,G_point_half_space_per_m,G_disk_half_space_per_m"
lines = [hdr] + [",".join(f"{r[k]:.6g}" for k in hdr.split(",")) for r in rows6]
(OUT / "part6_predictions.csv").write_text("\n".join(lines) + "\n")

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
ax = axs[0]
dd = geo["dist"] / mm
band = np.sqrt(U_num ** 2 + U_pos ** 2)
ax.fill_between(dd, G * (1 - band), G * (1 + band), color="#2a78d6", alpha=0.18, lw=0)
ax.loglog(dd, G, "o-", color="#2a78d6", lw=2, label="full model (scope ground = return)")
ax.loglog(dd, Ga3[2], "o--", color="#7b4fd0", lw=1.6, label="full model, separate far reference")
for (m, v), c, ls in zip(ideal.items(), ["#eb6834", "#1baf7a", "0.4"], ["-", "-", ":"]):
    ax.loglog(dd, v, ls, color=c, lw=1.6, label=m + (" (= point half space beyond ~2 mm)"
                                                      if m.startswith("disk") else ""))
ax.set_xlabel("distance from the stimulating disk (mm)")
ax.set_ylabel("G = phi sigma / I  (1/m)")
ax.set_title("Part 6: uniform tank (band: numerical + position, 95%)", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
ax = axs[1]
for r7, c in zip(part7, ["#2a78d6", "#eb6834"]):
    if r7["line_x_mm"]:
        gl = np.array(r7["G_line_per_m"])
        ax.plot(r7["line_x_mm"], gl - gl[-1], "o-", color=c, lw=2,
                label=f"stim at {r7['stim_offset_mm'][0]:.0f} mm")
ax.axvline(s7["agar_diameter_mm"] / 2, color="0.5", lw=1, ls="--")
ax.text(s7["agar_diameter_mm"] / 2 + 1, ax.get_ylim()[1] * 0.98 if ax.lines else 1, "agar boundary",
        fontsize=8, color="0.4", va="top")
ax.set_xlabel("position along the suggested line, x (mm from container centre)")
ax.set_ylabel("G - G(last position)  (1/m): offset-free")
ax.set_yscale("log")
ax.set_title(f"Part 7: line {s7.get('line_z_offset_mm', 0):g} mm above the tip, across the agar "
             f"boundary (sigma_in/out = {ratio0:.2f})", fontsize=10)
ax.grid(True, color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
fig.suptitle(f"Phantom blind predictions [{label}]" + (" -- QUICK, NOT FOR COMMITTING" if QUICK else ""))
fig.savefig(FIG / "phantom_predictions.png", dpi=200)
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
if not QUICK:
    print(f"NEXT: git add {OUT.relative_to(ROOT)} {FIG.relative_to(ROOT)} && git commit && "
          f"git tag {label} && git push --tags  -- BEFORE any measurement")
