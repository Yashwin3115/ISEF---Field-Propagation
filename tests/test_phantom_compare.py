"""Checks for the phantom analysis chain on SYNTHETIC data (no real
measurement involved): predictions -> fake measurements -> comparison.

1. disk_half_space reduces to the point half-space law far away and to the
   arctan axis formula on the axis (V1)
2. measurements made exactly from the prediction (+ small noise inside the
   stated uncertainties) are all covered, under BOTH referencing conditions
   (plan v16 steps 53-54); a 30% error is not
3. paired Wilcoxon (condition B) finds the full model better than the point
   formula when the data follow the full model
3b. the A - B difference test: covered, and flat across positions
3c. fd_ratio on an exact two-layer profile (straight lines with slopes in the
   ratio sigma_out / sigma_in) returns that ratio exactly; the gradient-ratio
   test covers exact data and rejects a 30% error
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
                              difference_test, gradient_ratio_test)
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

# synthetic committed predictions: condition B = 1/(2 pi r); A = B + 30 (return term)
d_mm = np.array([1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20.0])
GB = 1 / (2 * np.pi * d_mm * 1e-3)
GA = GB + 30
rows = [{"distance_mm": d, "G_A_per_m": ga, "U_num_A_rel": 0.02, "U_pos_A_rel": 0.01,
         "G_B_per_m": gb, "U_num_B_rel": 0.02, "U_pos_B_rel": 0.01, "U_refpos_B_rel": 0.005,
         "G_point_full_space_per_m": 1 / (4 * np.pi * d * 1e-3),
         "G_point_half_space_per_m": 1 / (2 * np.pi * d * 1e-3) * 1.15,
         "G_disk_half_space_per_m": 1 / (2 * np.pi * d * 1e-3) * 1.15} for d, ga, gb in zip(d_mm, GA, GB)]
pred = {"part6": {"rows": rows, "difference_A_minus_B": {"G_A_minus_B_per_m": 30.0, "U_num_rel": 0.02,
                                                        "U_refpos_rel": 0.01}}, "part7": []}
sessions = [{"session": 1.0, "sigma_start": 1.55, "sigma_end": 1.56, "sigma_rel_sd": 0.005},
            {"session": 2.0, "sigma_start": 1.50, "sigma_end": 1.50, "sigma_rel_sd": 0.005}]
rng = np.random.default_rng(1)
def make(scale=1.0):
    meas = []
    for s in sessions:
        sig = 0.5 * (s["sigma_start"] + s["sigma_end"])
        for k in range(len(d_mm)):
            for I in (50e-6, 100e-6):
                for cond, g in (("A", GA[k]), ("B", GB[k])):
                    V = I * g / sig * scale * (1 + rng.normal(0, 0.003))
                    meas.append({"part": 6.0, "session": s["session"], "position": float(k), "condition": cond,
                                 "stim_index": 0.0, "pulse_width_us": 100.0, "current_A": I,
                                 "current_rel_sd": 0.005, "plateau_V": V, "noise_floor_V": 2e-6})
    return meas

# 2
r_ok = predict_rows(make(1.0), sessions, pred)
r_bad = predict_rows(make(1.3), sessions, pred)
cA = coverage([r for r in r_ok if r["condition"] == "A"])[0]
cB = coverage([r for r in r_ok if r["condition"] == "B"])[0]
c_bad = coverage(r_bad)[0]
check("coverage rule, both conditions", cA == 1.0 and cB == 1.0 and c_bad == 0.0,
      f"exact data A {cA*100:.0f}%, B {cB*100:.0f}% covered; 30% off {c_bad*100:.0f}%")

# 3
B_ok = [r for r in r_ok if r["condition"] == "B"]
w = wilcoxon_vs_full(B_ok)
pw = w["point, full space"]["p"]
check("paired Wilcoxon (condition B)", pw < 1e-6, f"full vs point full space p = {pw:.1e} (n = {w['point, full space']['n']})")

# 3b
dt, flat = difference_test(r_ok, sessions, pred)
spread = max(f["rel_spread"] for f in flat.values())
check("A - B difference test", len(dt) == 40 and all(d["covered"] for d in dt) and spread < 0.25,
      f"{sum(d['covered'] for d in dt)}/{len(dt)} covered; spread across positions {spread*100:.1f}% "
      f"(noise only; predicted 0)")

# 3c exact two-layer profile: inside slope -s/ratio, outside -s
ratio_true = 3.6                                     # sigma_in / sigma_out
dist = np.array([5, 15, 25, 30, 35, 37.5, 42.5, 45, 50, 55, 60, 65.0])
R_if = 40.0
def prof(r):
    return np.where(r < R_if, -(r - R_if) / ratio_true, -(r - R_if)) * 1.0 + 100.0
R_fd, idx = fd_ratio(dist * 1e-3, prof(dist), R_if * 1e-3)
p7 = {"traverse_distances_mm": dist.tolist(), "ratio_in_out_nominal": ratio_true,
      "G_per_m": prof(dist).tolist(), "U_num_rel": [0.01] * 12, "U_pos_rel": [0.0] * 12,
      "dlnG_dln_ratio": [0.0] * 12,
      "gradient_ratio": {"points_used_mm": dist[list(idx)].tolist(), "fd_prediction": float(R_fd),
                         "U_num_rel": 0.01, "U_pos_rel": 0.03, "dlnR_dln_ratio": -1.0}}
pred7 = {"part6": pred["part6"], "part7": [p7]}
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
      abs(R_fd - 1 / ratio_true) < 1e-12 and g_ok[0]["covered"] and not g_bad[0]["covered"],
      f"fd ratio {R_fd:.6f} = 1/{ratio_true}; exact data covered, 30% error rejected "
      f"(measured {g_bad[0]['R_meas']:.4f} vs {g_bad[0]['R_pred']:.4f})")

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
