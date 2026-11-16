"""Checks for the phantom analysis chain on SYNTHETIC data (no real
measurement involved): predictions -> fake measurements -> comparison.

1. disk_half_space reduces to the point half-space law far away and to the
   arctan axis formula on the axis (V1)
2. measurements made exactly from the prediction (+ small noise inside the
   stated uncertainties) are all covered, under BOTH referencing conditions
   and in every layout; a 30% error is not
3. the committed criteria: k_min, and the "simple model fails" rule finds the
   point formula worse when the data follow the full model, and does NOT
   when the data follow the point formula
3b. the A - B difference test: covered, and flat across positions
3c. fd_ratio on an exact two-layer profile returns sigma_out / sigma_in
   exactly; the gradient-ratio test covers exact data, rejects a 30% error,
   and excludes the "no interface" ratio
3d. ratios between layouts (plate ratio, return ratio): exact data covered,
   a constant-2 world rejected; two salines collapse, a 10% conductivity
   error does not
4. the decay-exponent fit recovers a known exponent; bootstrap CI contains it
5. plateau() recovers a pulse's first-phase level from a noisy biphasic trace
6. fit_randles recovers known R_s, R_ct, CPE parameters, also with a wiring
   baseline in parallel
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from phantom import disk_half_space, point_half_space, fd_ratio
from phantom_analysis import (predict_rows, coverage, wilcoxon_vs_full, decay_exponent,
                              difference_test, gradient_ratio_test, ratio_test, sigma_collapse_test,
                              reciprocity)
from phantom_criteria import k_min, closer_test, resolve_criteria, verdict
from scope import plateau, randles, fit_randles

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

# 1
a = 0.25e-3
far = disk_half_space(np.array([20e-3]), np.array([0.0]), a)[0] / point_half_space(20e-3) - 1
axis = disk_half_space(np.array([0.0]), np.array([1e-3]), a)[0] / (np.arctan(a / 1e-3) / (2 * np.pi * a)) - 1
check("disk formula limits", abs(far) < 1e-3 and abs(axis) < 1e-9, f"far {far:.1e}, axis {axis:.1e}")

# synthetic committed predictions. "Full model" world: condition B = 0.9/(2 pi r) - 2
# (walls + reference), A = B + 30 (return term). Layout E2 (no plate) = half of E1 far
# away and 0.6 of it at 1 mm; layout E3 (ring return) = E1 x (1 - d/40).
d_mm = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20.0])
half = 1 / (2 * np.pi * d_mm * 1e-3)
def layout_rows(GB, off, band=0.03):
    GA = GB + off
    return [{"distance_mm": float(d), "G_A_per_m": float(ga), "G_A_lo_per_m": float(ga * (1 - band)),
             "G_A_hi_per_m": float(ga * (1 + band)), "G_B_per_m": float(gb),
             "G_B_lo_per_m": float(gb * (1 - band)), "G_B_hi_per_m": float(gb * (1 + band)),
             "reposition_ratio_A_band": [1 - band, 1 + band], "reposition_ratio_B_band": [1 - band, 1 + band],
             "G_point_full_space_per_m": float(h / 2), "G_point_half_space_per_m": float(h),
             "G_disk_half_space_per_m": float(h),
             "G_point_half_space_referenced_per_m": float(h - 1.0)} for d, ga, gb, h in zip(d_mm, GA, GB, half)]
GB1 = 0.9 * half - 2
GB2 = GB1 / (2.0 - 0.35 * np.exp(-(d_mm - 1) / 1.5))
GB3 = GB1 * (1 - d_mm / 40)
EXP = {"E1_distance": {"rows": layout_rows(GB1, 30.0), "difference_A_minus_B":
                       {"G_A_minus_B_per_m": 30.0, "G_lo_per_m": 29.0, "G_hi_per_m": 31.0}},
       "E2_no_plate": {"rows": layout_rows(GB2, 33.0), "setup_part6": {"support": {"type": "post"}},
                       "difference_A_minus_B": {"G_A_minus_B_per_m": 33.0, "G_lo_per_m": 32.0, "G_hi_per_m": 34.0}},
       "E3_return_ring": {"rows": layout_rows(GB3, 5.0), "difference_A_minus_B":
                          {"G_A_minus_B_per_m": 5.0, "G_lo_per_m": 4.8, "G_hi_per_m": 5.2}}}
def table(num, den, band=0.04):
    return [{"distance_mm": float(d), "ratio": float(a / b), "lo": float(a / b * (1 - band)),
             "hi": float(a / b * (1 + band)), "usable": True} for d, a, b in zip(d_mm, num, den)]
derived = {"plate_ratio": {"condition_B": table(GB1, GB2)},
           "return_ratio": {"E3_return_ring": table(GB3, GB1)}}
setup = {"uncertainties": {"sigma_meter_rel_sd": 0.005, "sigma_temp_coeff_per_C": 0.02, "temperature_sd_C": 0.2,
                           "current_rel_sd": 0.005, "scope_gain_rel_sd": 0.005}}
pred = {"experiments": EXP, "part6": EXP["E1_distance"], "derived": derived, "part7": [], "setup": setup}
crit = resolve_criteria(EXP, derived, [], setup)
sessions = [{"session": 1.0, "sigma_start": 1.55, "sigma_end": 1.56, "sigma_rel_sd": 0.005},
            {"session": 2.0, "sigma_start": 0.80, "sigma_end": 0.80, "sigma_rel_sd": 0.005}]
rng = np.random.default_rng(1)
def make(scale=1.0, world="full", sigma_error=1.0):
    meas = []
    for s in sessions:
        sig = 0.5 * (s["sigma_start"] + s["sigma_end"])
        for lay, ex in EXP.items():
            for k, row in enumerate(ex["rows"]):
                for cond in ("A", "B"):
                    g = row[f"G_{cond}_per_m"]
                    if world == "point" and cond == "B":
                        g = row["G_point_half_space_referenced_per_m"] * (0.5 if lay == "E2_no_plate" else 1.0)
                    err = sigma_error if s["session"] == 2.0 else 1.0
                    V = 1e-4 * g / sig * scale * err + rng.normal(0, 3e-7)
                    meas.append({"part": 6.0, "session": s["session"], "layout": lay, "position": float(k),
                                 "condition": cond, "stim_index": 0.0, "pulse_width_us": 100.0,
                                 "current_A": 1e-4, "current_rel_sd": 0.005, "plateau_V": V,
                                 "noise_floor_V": 3e-7})
    return meas

# 2
r_ok = predict_rows(make(1.0), sessions, pred)
r_bad = predict_rows(make(1.3), sessions, pred)
cA = coverage([r for r in r_ok if r["condition"] == "A"])[0]
cB = coverage([r for r in r_ok if r["condition"] == "B"])[0]
c_bad = coverage(r_bad)[0]
check("coverage rule, every layout, both conditions", cA == 1.0 and cB == 1.0 and c_bad == 0.0,
      f"exact data A {cA*100:.0f}%, B {cB*100:.0f}% covered ({len(r_ok)} readings); 30% off {c_bad*100:.0f}%")

# 3 committed criteria
kk = (k_min(13), k_min(26), k_min(5), k_min(1))
c1 = crit["experiments"]["E1_distance"]["simple_models_condition_B"]["point, half space, referenced"]
def closer(rows_, name="point, half space, referenced"):
    sel = [r for r in rows_ if r["layout"] == "E1_distance" and r["condition"] == "B"
           and int(r["position"]) in c1["discriminating_positions"]]
    m = np.array([r["plateau_V"] for r in sel])
    return closer_test(np.abs(m - np.array([r["V_models"]["full model"] for r in sel])),
                       np.abs(m - np.array([r["V_models"][name] for r in sel])))
ct_full = closer(r_ok)
ct_point = closer(predict_rows(make(1.0, "point"), sessions, pred))
check("pass criteria (k_min; simple model fails only when it should)",
      kk == (11, 23, 4, 1) and c1["testable"] and ct_full["simple_model_fails"]
      and not ct_point["simple_model_fails"],
      f"k_min 13 -> {kk[0]}, 26 -> {kk[1]}, 5 -> {kk[2]}; {len(c1['discriminating_positions'])} discriminating "
      f"positions; full-model world: point fails (closer {ct_full['full_closer_fraction']*100:.0f}%, "
      f"p {ct_full['p']:.1e}); point-source world: closer {ct_point['full_closer_fraction']*100:.0f}% -> no failure")

# 3b
dt, flat = difference_test(r_ok, sessions, pred)
spread = max(f["rel_spread"] for f in flat.values())
allowed = min(f["allowed_rel_spread"] for f in flat.values())
check("A - B difference test", len(dt) == 60 and all(d["covered"] for d in dt) and spread < allowed,
      f"{sum(d['covered'] for d in dt)}/{len(dt)} covered; spread across positions {spread*100:.2f}% "
      f"(noise only; allowed {allowed*100:.1f}%)")

# 3c exact two-layer profile: inside slope -s/ratio, outside -s
ratio_true = 3.6                                     # sigma_in / sigma_out
dist = np.array([5, 15, 25, 30, 35, 37.5, 42.5, 45, 50, 55, 60, 65.0])
R_if = 40.0
def prof(r):
    return np.where(r < R_if, -(r - R_if) / ratio_true, -(r - R_if)) * 1.0 + 100.0
R_fd, idx = fd_ratio(dist * 1e-3, prof(dist), R_if * 1e-3)
p7 = {"traverse_distances_mm": dist.tolist(), "ratio_in_out_nominal": ratio_true,
      "G_per_m": prof(dist).tolist(), "G_lo_per_m": (prof(dist) * 0.98).tolist(),
      "G_hi_per_m": (prof(dist) * 1.02).tolist(), "dlnG_dln_ratio": [0.0] * 12,
      "gradient_ratio": {"points_used_mm": dist[list(idx)].tolist(), "fd_prediction": float(R_fd),
                         "band_position_and_grid": [float(R_fd) * 0.96, float(R_fd) * 1.04],
                         "uniform_medium_ratio": 0.8, "U_num_rel": 0.01, "U_pos_rel": 0.03,
                         "dlnR_dln_ratio": -1.0}}
pred7 = dict(pred, part7=[p7])
s7 = [{"session": 3.0, "sigma_start": 0.8, "sigma_end": 0.8, "sigma_rel_sd": 0.003,
       "sigma_in_start": 2.88, "sigma_in_end": 2.88}]
def make7(scale_in=1.0):
    m = []
    for k, r in enumerate(dist):
        g = prof(np.array([r]))[0]
        if r < R_if:
            g = 100.0 + (g - 100.0) * scale_in
        m.append({"part": 7.0, "session": 3.0, "position": float(k), "condition": "A", "stim_index": 0.0,
                  "pulse_width_us": 100.0, "current_A": 1e-4, "current_rel_sd": 0.003,
                  "plateau_V": 1e-4 * g / 0.8, "noise_floor_V": 0.0})
    return m
g_ok = gradient_ratio_test(predict_rows(make7(1.0), s7, pred7), s7, pred7)
g_bad = gradient_ratio_test(predict_rows(make7(1.3), s7, pred7), s7, pred7)
check("gradient ratio (exact two-layer profile)",
      abs(R_fd - 1 / ratio_true) < 1e-12 and g_ok[0]["covered"] and not g_bad[0]["covered"]
      and g_ok[0]["uniform_medium_excluded"],
      f"fd ratio {R_fd:.6f} = 1/{ratio_true}; exact data covered, 30% error rejected "
      f"(measured {g_bad[0]['R_meas']:.4f} vs {g_bad[0]['R_pred']:.4f}); 'no interface' (0.8) excluded")

# 3d ratios between layouts; two salines
pr = ratio_test(r_ok, pred, derived["plate_ratio"]["condition_B"], "E1_distance", "E2_no_plate", 2.0)
rr = ratio_test(r_ok, pred, derived["return_ratio"]["E3_return_ring"], "E3_return_ring", "E1_distance", 1.0)
pr_pt = ratio_test(predict_rows(make(1.0, "point"), sessions, pred), pred,
                   derived["plate_ratio"]["condition_B"], "E1_distance", "E2_no_plate", 2.0)
cp = crit["derived"]["plate_ratio"]
sel = lambda t: [q for q in t if q["position"] in cp["discriminating_positions"]]
v_full = closer_test([q["e_full"] for q in sel(pr)], [q["e_simple"] for q in sel(pr)])
v_pt = closer_test([q["e_full"] for q in sel(pr_pt)], [q["e_simple"] for q in sel(pr_pt)])
sc = sigma_collapse_test(r_ok, pred)
sc_bad = sigma_collapse_test(predict_rows(make(1.0, sigma_error=1.10), sessions, pred), pred)
check("layout ratios and conductivity collapse",
      all(q["covered"] for q in pr) and all(q["covered"] for q in rr) and len(pr) == 10
      and v_full["simple_model_fails"] and not v_pt["simple_model_fails"]
      and len(sc) == 60 and all(q["covered"] for q in sc) and not any(q["covered"] for q in sc_bad),
      f"plate ratio {pr[0]['R_meas']:.2f}..{pr[-1]['R_meas']:.2f} and return ratio covered; 'always 2' fails "
      f"in the full-model world, not in a point-source world; collapse {len(sc)}/{len(sc)} "
      f"(median {np.median([q['collapse_ratio'] for q in sc]):.3f}); 10% sigma error -> "
      f"{sum(q['covered'] for q in sc_bad)}/{len(sc_bad)} covered")
rc = reciprocity([{"session": 1.0, "layout": "E1_distance", "distance_mm": 5.0, "V_forward": 1e-3,
                   "V_swapped": 1.01e-3, "noise_floor_V": 2e-6},
                  {"session": 1.0, "layout": "E1_distance", "distance_mm": 5.0, "V_forward": 1e-3,
                   "V_swapped": 1.06e-3, "noise_floor_V": 2e-6}])
check("reciprocity tolerance", rc[0]["pass"] and not rc[1]["pass"],
      f"1% difference passes, 6% does not (allowed {rc[0]['allowed']*100:.1f}%)")

# 4 data with a known exponent 1.2 (no offset)
meas = []
for s in (1.0, 2.0, 3.0):
    for k, d in enumerate(d_mm):
        meas.append({"session": s, "distance_mm": d, "current_A": 50e-6,
                     "plateau_V": 50e-6 * d ** -1.2 * (1 + rng.normal(0, 0.01))})
de = decay_exponent(meas, n_boot=300)
check("decay exponent + bootstrap", abs(de["exponent"] - 1.2) < 0.02 and de["ci95"][0] <= 1.2 <= de["ci95"][1],
      f"n = {de['exponent']:.3f}, CI [{de['ci95'][0]:.3f}, {de['ci95'][1]:.3f}]")

# 5 biphasic trace: 0 baseline (+0.3 mV offset), +2 mV for 200 us, -2 mV for 200 us
t = np.arange(0, 1e-3, 1e-6)
v = 0.3e-3 + np.where((t >= 0.2e-3) & (t < 0.4e-3), 2e-3, 0) + np.where((t >= 0.4e-3) & (t < 0.6e-3), -2e-3, 0)
v = v + rng.normal(0, 20e-6, len(t))
pv, sd, info = plateau(t, v)
check("plateau extraction", abs(pv - 2e-3) < 1e-5 and abs(info["phase_width_s"] - 200e-6) < 5e-6,
      f"{pv*1e3:.4f} mV (true 2.0000), phase {info['phase_width_s']*1e6:.0f} us")

# 6 Randles + wiring baseline
f = np.array([10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5])
Ztrue = randles(f, 600.0, 2e6, 2e-7, 0.85)
Zw = 1 / (1j * 2 * np.pi * f * 20e-12)                     # 20 pF stray, in parallel
Zm = 1 / (1 / Ztrue + 1 / Zw)
fit = fit_randles(f, Zm, Zw)
check("Randles fit", abs(fit["R_s_ohm"] / 600 - 1) < 1e-3 and abs(fit["alpha"] - 0.85) < 1e-3,
      f"R_s {fit['R_s_ohm']:.2f} (600), alpha {fit['alpha']:.4f} (0.85), Q {fit['Q']:.3g} (2e-7)")

print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
