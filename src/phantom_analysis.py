"""Phantom comparison (research plan v16 Parts 6-7 + the Nov 15 experiment
list): measured vs COMMITTED predictions. Used by runs/phantom_compare.py;
tested by tests/test_phantom_compare.py on synthetic data.

Inputs (CSV, templates in notes/):
  measurements:  one row per (session, layout, position, condition, pulse width, amplitude):
      part, session, layout, position, condition, stim_index, pulse_width_us, current_A,
      current_rel_sd, plateau_V, plateau_V_sd, n_traces, noise_floor_V
      part 6: layout = the experiment name in the setup file (E1_distance, ...);
              position = index into the committed distances;
              condition = A (scope ground at the return) or B (far reference wire)
      part 7: position = index along the committed traverse; condition = A
  sessions:      session, part, sigma_start, sigma_end, sigma_rel_sd,
                 temp_start_C, temp_end_C, sigma_in_start, sigma_in_end
                 (sigma_in_*: agar, part 7 only; sigma_* is the saline)
  reciprocity:   session, layout, distance_mm, V_forward, V_swapped, noise_floor_V

Tests applied (all fixed before measurement, in predictions.json and
pass_criteria.json):
  coverage of every reading by its committed band (each layout, A and B, traverse)
  the A - B difference: predicted to be the SAME at every position
  the plate ratio (plate / no plate) and the return ratios (other return / far wall)
  two saline concentrations: readings x sigma / I collapse
  the gradient ratio across the agar interface

Statistical units (plan 5.12): one row = one measurement (N averaged traces
at one position in one session). Paired tests pair the SAME measurement
under different models. The decay-exponent bootstrap resamples sessions,
then positions within each session -- never samples.
"""
import csv
import numpy as np
from scipy import stats

# models that assume the return is at infinity are compared with condition B
MODELS_B = {"full model": "G_B_per_m",
            "point, full space": "G_point_full_space_per_m",
            "point, half space": "G_point_half_space_per_m",
            "disk, half space": "G_disk_half_space_per_m",
            "point, full space, referenced": "G_point_full_space_referenced_per_m",
            "point, half space, referenced": "G_point_half_space_referenced_per_m"}


def read_csv(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if any((v or "").strip() for v in r.values())]
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            k = k.strip(); v = (v or "").strip()
            try:
                d[k] = float(v)
            except ValueError:
                d[k] = v
        out.append(d)
    return out


def session_sigma(s, inner=False):
    """Mean conductivity of a session and its relative standard uncertainty,
    drift included: the value may lie anywhere between start and end
    (uniform distribution -> half-range / sqrt(3)), plus the meter's spread."""
    a, b = (s["sigma_in_start"], s["sigma_in_end"]) if inner else (s["sigma_start"], s["sigma_end"])
    m = 0.5 * (a + b)
    drift = abs(b - a) / 2 / m / np.sqrt(3)
    return m, float(np.hypot(s.get("sigma_rel_sd", 0.0) or 0.0, drift)), abs(b - a) / m


def _cond(m):
    c = str(m.get("condition", "A") or "A").strip().upper()
    if c not in ("A", "B"):
        raise ValueError(f"condition must be A or B, got {c!r}")
    return c


def layout_of(m, pred):
    lay = str(m.get("layout", "") or "").strip()
    if not lay or lay not in pred.get("experiments", {}):
        if lay and "experiments" in pred:
            raise ValueError(f"layout {lay!r} is not in the committed predictions "
                             f"({list(pred['experiments'])})")
        lay = next(iter(pred["experiments"])) if "experiments" in pred else "baseline"
    return lay


def _exp(pred, lay):
    return pred["experiments"][lay] if "experiments" in pred else pred["part6"]


def gain_sd(pred):
    return float(pred.get("setup", {}).get("uncertainties", {}).get("scope_gain_rel_sd", 0.0) or 0.0)


def predict_rows(meas, sessions, pred):
    """Attach V_pred (each applicable model) and the committed band to every
    row. Rule (predictions.json): covered if the reading lies between the
    band limits I G_lo / sigma and I G_hi / sigma, each half-width widened in
    quadrature by 2 sqrt(u_sigma^2 + u_I^2 + u_gain^2) |V_pred|, then by
    2 x the reading's noise floor."""
    sess = {str(s["session"]): s for s in sessions}
    u_gain = gain_sd(pred)
    out = []
    for m in meas:
        s = sess[str(m["session"])]
        sig, u_sig, drift = session_sigma(s)
        u_I = m.get("current_rel_sd", 0.0) or 0.0
        I = m["current_A"]
        r = dict(m)
        r["condition"] = _cond(m)
        u_extra = 0.0
        if int(m["part"]) == 6:
            lay = layout_of(m, pred)
            r["layout"] = lay
            row = _exp(pred, lay)["rows"][int(m["position"])]
            r["distance_mm"] = row["distance_mm"]
            c = r["condition"]
            G, G_lo, G_hi = row[f"G_{c}_per_m"], row[f"G_{c}_lo_per_m"], row[f"G_{c}_hi_per_m"]
            if c == "A":
                r["V_models"] = {"full model": I * G / sig}
            else:
                r["V_models"] = {name: I * row[k] / sig for name, k in MODELS_B.items() if k in row}
            # the prediction is "clearly not zero": used to decide where relative errors mean anything
            r["well_above_zero"] = bool(abs(G) >= 3 * (G_hi - G_lo))
        else:
            p7 = pred["part7"][int(m.get("stim_index", 0) or 0)]
            sig_in, u_in, _ = session_sigma(s, inner=True)
            ratio = sig_in / sig
            k = int(float(m["position"]))
            dl = p7["dlnG_dln_ratio"][k]
            r["distance_mm"] = p7["traverse_distances_mm"][k]
            fac = (ratio / p7["ratio_in_out_nominal"]) ** dl
            G, G_lo, G_hi = p7["G_per_m"][k] * fac, p7["G_lo_per_m"][k] * fac, p7["G_hi_per_m"][k] * fac
            r["V_models"] = {"full model": I * G / sig}
            u_extra = np.hypot(u_sig, u_in) * abs(dl)
            r["well_above_zero"] = True
        Vp = I * G / sig
        mult = 2 * np.sqrt(u_sig ** 2 + u_I ** 2 + u_gain ** 2 + u_extra ** 2) * abs(Vp)
        noise = 2 * (m.get("noise_floor_V", 0.0) or 0.0)
        lo, hi = sorted((I * G_lo / sig, I * G_hi / sig))
        r["V_pred"] = Vp
        r["band_lo_V"] = float(Vp - np.hypot(Vp - lo, mult) - noise)
        r["band_hi_V"] = float(Vp + np.hypot(hi - Vp, mult) + noise)
        r["band_V"] = float(max(Vp - r["band_lo_V"], r["band_hi_V"] - Vp))
        r["U_rel"] = float(r["band_V"] / abs(Vp)) if Vp else float("inf")
        r["covered"] = bool(r["band_lo_V"] <= m["plateau_V"] <= r["band_hi_V"])
        r["sigma_used"] = sig
        r["u_sigma"] = u_sig
        r["sigma_drift_rel"] = drift
        out.append(r)
    return out


def difference_test(rows, sessions, pred):
    """Part 6: V_A - V_B at each (session, layout, position, pulse width,
    current) pair, against the committed position-independent difference."""
    u_gain = gain_sd(pred)
    pairs = {}
    for r in rows:
        if int(r["part"]) != 6:
            continue
        key = (str(r["session"]), r["layout"], int(r["position"]), r.get("pulse_width_us"),
               round(r["current_A"], 9))
        pairs.setdefault(key, {})[r["condition"]] = r
    out = []
    for key, pr in pairs.items():
        if "A" not in pr or "B" not in pr:
            continue
        a, b = pr["A"], pr["B"]
        D = _exp(pred, key[1])["difference_A_minus_B"]
        sig, u_sig = a["sigma_used"], a["u_sigma"]
        I = 0.5 * (a["current_A"] + b["current_A"])
        u_I = max(a.get("current_rel_sd", 0) or 0, b.get("current_rel_sd", 0) or 0)
        Vp = I * D["G_A_minus_B_per_m"] / sig
        lo, hi = sorted((I * D["G_lo_per_m"] / sig, I * D["G_hi_per_m"] / sig))
        mult = 2 * np.sqrt(u_sig ** 2 + u_I ** 2 + u_gain ** 2) * abs(Vp)
        noise = 2 * np.hypot(a.get("noise_floor_V", 0) or 0, b.get("noise_floor_V", 0) or 0)
        Vm = a["plateau_V"] - b["plateau_V"]
        blo, bhi = Vp - np.hypot(Vp - lo, mult) - noise, Vp + np.hypot(hi - Vp, mult) + noise
        out.append({"session": key[0], "layout": key[1], "position": key[2], "distance_mm": a["distance_mm"],
                    "pulse_width_us": key[3], "current_A": I, "V_meas": Vm, "V_pred": Vp,
                    "band_lo_V": float(blo), "band_hi_V": float(bhi),
                    "covered": bool(blo <= Vm <= bhi), "V_meas_over_I": Vm / I,
                    "noise_rel": float(noise / 2 / abs(Vm)) if Vm else None})
    # flatness: is the measured difference the same at every position?
    flat = {}
    for key in sorted(set((o["session"], o["layout"]) for o in out)):
        sel = [o for o in out if (o["session"], o["layout"]) == key]
        v = np.array([o["V_meas_over_I"] for o in sel])
        nz = max((o["noise_rel"] or 0) for o in sel)
        flat[f"{key[0]} / {key[1]}"] = {
            "n": len(v), "rel_spread": float(np.ptp(v) / abs(np.mean(v))) if len(v) > 1 else None,
            "allowed_rel_spread": float(np.hypot(0.02, 2 * np.sqrt(2) * 2 * nz))}
    return out, flat


def _normalized(rows, layout, condition="B"):
    """V sigma / I for every part-6 reading of one layout and condition, keyed
    by (position, pulse width); readings from several sessions are averaged."""
    acc = {}
    for r in rows:
        if int(r["part"]) != 6 or r["layout"] != layout or r["condition"] != condition:
            continue
        key = (int(r["position"]), r.get("pulse_width_us"))
        x = r["plateau_V"] * r["sigma_used"] / r["current_A"]
        nz = (r.get("noise_floor_V", 0) or 0) / abs(r["plateau_V"]) if r["plateau_V"] else 0.0
        acc.setdefault(key, []).append((x, nz, r["u_sigma"], r.get("current_rel_sd", 0) or 0,
                                        str(r["session"]), r["distance_mm"]))
    out = {}
    for key, v in acc.items():
        a = np.array([q[:4] for q in v], float)
        n = len(v)
        out[key] = {"x": float(a[:, 0].mean()), "noise_rel": float(np.sqrt(np.mean(a[:, 1] ** 2) / n)),
                    "u_sigma": float(a[:, 2].max()), "u_I": float(a[:, 3].max()),
                    "sessions": sorted(set(q[4] for q in v)), "distance_mm": v[0][5]}
    return out


def ratio_test(rows, pred, table, num_layout, den_layout, simple_value, condition="B"):
    """A committed ratio of two layouts' readings at the same distance (the
    plate ratio, a return ratio): measured ratio of V sigma / I, the committed
    band (independent probe placements, grid) widened in quadrature by
    2 sqrt(u_sigma terms + 2 u_I^2 + 2 u_gain^2) and by both noise floors.
    The conductivity terms drop out when both readings come from the same
    session (same saline). Also returns each reading's distance from the full
    model and from the simple model's constant value."""
    u_gain = gain_sd(pred)
    N, D = _normalized(rows, num_layout, condition), _normalized(rows, den_layout, condition)
    out = []
    for key in sorted(set(N) & set(D), key=lambda k: (k[0], str(k[1]))):
        q = table[key[0]]
        if not q["usable"]:
            continue
        n, d = N[key], D[key]
        Rm = n["x"] / d["x"]
        same = n["sessions"] == d["sessions"]
        w = 2 * np.sqrt((0 if same else n["u_sigma"] ** 2 + d["u_sigma"] ** 2) + n["u_I"] ** 2
                        + d["u_I"] ** 2 + 2 * u_gain ** 2 + n["noise_rel"] ** 2 + d["noise_rel"] ** 2)
        R = q["ratio"]
        lo, hi = R - np.hypot(R - q["lo"], w * R), R + np.hypot(q["hi"] - R, w * R)
        out.append({"position": key[0], "distance_mm": q["distance_mm"], "pulse_width_us": key[1],
                    "R_meas": float(Rm), "R_pred": float(R), "band": [float(lo), float(hi)],
                    "covered": bool(lo <= Rm <= hi), "e_full": float(abs(Rm - R)),
                    "e_simple": float(abs(Rm - simple_value)), "same_session": same})
    return out


def sigma_collapse_test(rows, pred, min_sigma_ratio=1.1):
    """Two saline concentrations (experiment e): the same layout, position and
    condition read in sessions whose conductivities differ by more than
    min_sigma_ratio. G does not depend on sigma, so (V sigma / I) must be the
    same. Band: the committed repositioning band (the probe is put back between
    sessions) widened by both sessions' conductivity, current, gain and noise."""
    u_gain = gain_sd(pred)
    groups = {}
    for r in rows:
        if int(r["part"]) != 6:
            continue
        key = (r["layout"], int(r["position"]), r["condition"], r.get("pulse_width_us"))
        groups.setdefault(key, {}).setdefault(str(r["session"]), []).append(r)
    out = []
    for key, by_s in groups.items():
        ss = sorted(by_s, key=lambda k: by_s[k][0]["sigma_used"])
        if len(ss) < 2:
            continue
        lo_s, hi_s = ss[0], ss[-1]
        if by_s[hi_s][0]["sigma_used"] / by_s[lo_s][0]["sigma_used"] < min_sigma_ratio:
            continue
        def norm(rr):
            x = np.mean([r["plateau_V"] * r["sigma_used"] / r["current_A"] for r in rr])
            nz = np.sqrt(np.mean([((r.get("noise_floor_V", 0) or 0) / abs(r["plateau_V"])) ** 2 for r in rr]) / len(rr))
            return x, nz, rr[0]["u_sigma"], max((r.get("current_rel_sd", 0) or 0) for r in rr)
        xh, nh, uh, ih = norm(by_s[hi_s]); xl, nl, ul, il = norm(by_s[lo_s])
        c = xh / xl
        row = _exp(pred, key[0])["rows"][key[1]]
        rep = row.get(f"reposition_ratio_{key[2]}_band", [1.0, 1.0])
        w = 2 * np.sqrt(uh ** 2 + ul ** 2 + ih ** 2 + il ** 2 + 2 * u_gain ** 2 + nh ** 2 + nl ** 2)
        if not all(np.isfinite(rep)) or not by_s[hi_s][0]["well_above_zero"]:
            continue
        lo, hi = 1 - np.hypot(1 - rep[0], w), 1 + np.hypot(rep[1] - 1, w)
        out.append({"layout": key[0], "position": key[1], "distance_mm": row["distance_mm"],
                    "condition": key[2], "pulse_width_us": key[3], "sessions": [lo_s, hi_s],
                    "sigma": [by_s[lo_s][0]["sigma_used"], by_s[hi_s][0]["sigma_used"]],
                    "collapse_ratio": float(c), "band": [float(lo), float(hi)],
                    "covered": bool(lo <= c <= hi)})
    return out


def gradient_ratio_test(rows, sessions, pred):
    """Part 7: the gradient ratio from the four committed traverse points, per
    (session, stim position, pulse width, current), against the committed
    finite-difference prediction corrected to the measured conductivities."""
    sess = {str(s["session"]): s for s in sessions}
    groups = {}
    for r in rows:
        if int(r["part"]) != 7:
            continue
        key = (str(r["session"]), int(r.get("stim_index", 0) or 0), r.get("pulse_width_us"),
               round(r["current_A"], 9))
        groups.setdefault(key, {})[int(float(r["position"]))] = r
    out = []
    for key, g in groups.items():
        p7 = pred["part7"][key[1]]
        gr = p7["gradient_ratio"]
        dist = np.array(p7["traverse_distances_mm"])
        idx = [int(np.argmin(np.abs(dist - x))) for x in gr["points_used_mm"]]
        if not all(i in g for i in idx):
            continue
        pts = dist[idx] * 1e-3
        V = np.array([g[i]["plateau_V"] for i in idx])
        def ratio_of(Vv):
            return ((Vv[1] - Vv[0]) / (pts[1] - pts[0])) / ((Vv[3] - Vv[2]) / (pts[3] - pts[2]))
        R_meas = ratio_of(V)
        s = sess[key[0]]
        sig, u_sig, _ = session_sigma(s)
        sig_in, u_in, _ = session_sigma(s, inner=True)
        ratio = sig_in / sig
        R_pred = gr["fd_prediction"] * (ratio / p7["ratio_in_out_nominal"]) ** gr["dlnR_dln_ratio"]
        noise = np.array([g[i].get("noise_floor_V", 0) or 0 for i in idx])
        U_noise = float(np.sqrt(sum((ratio_of(V + np.eye(4)[k] * noise[k]) / R_meas - 1) ** 2
                                    for k in range(4)))) if np.any(noise) else 0.0
        u_ratio = np.hypot(u_sig, u_in) * abs(gr["dlnR_dln_ratio"])
        # committed band (positions + grid) scaled to the measured conductivities,
        # widened by the session's noise and conductivity-ratio uncertainty
        fac = R_pred / gr["fd_prediction"]
        blo, bhi = (np.array(gr["band_position_and_grid"]) * fac if "band_position_and_grid" in gr else
                    R_pred * np.array([1 - np.hypot(gr["U_num_rel"], gr["U_pos_rel"]),
                                       1 + np.hypot(gr["U_num_rel"], gr["U_pos_rel"])]))
        w = np.hypot(2 * U_noise, 2 * u_ratio) * abs(R_pred)
        lo, hi = R_pred - np.hypot(R_pred - blo, w), R_pred + np.hypot(bhi - R_pred, w)
        uni = gr.get("uniform_medium_ratio")
        out.append({"session": key[0], "stim_index": key[1], "pulse_width_us": key[2],
                    "current_A": key[3], "R_meas": float(R_meas), "R_pred": float(R_pred),
                    "band": [float(lo), float(hi)],
                    "R_ideal_measured_sigmas": float(sig / sig_in),
                    "gap_fd_vs_ideal": float(R_pred / (sig / sig_in) - 1),
                    "U_rel": float(max(R_pred - lo, hi - R_pred) / abs(R_pred)), "U_noise_rel": U_noise,
                    "covered": bool(lo <= R_meas <= hi),
                    "uniform_medium_ratio": uni,
                    "uniform_medium_excluded": (None if uni is None else
                                                bool(not (R_meas - (R_pred - lo) <= uni <= R_meas + (hi - R_pred))))})
    return out


def coverage(rows):
    c = np.array([r["covered"] for r in rows])
    return float(c.mean()) if len(c) else np.nan, int(c.sum()), len(c)


def errors_by_model(rows):
    """Relative error and RMSE per model (part 6 rows)."""
    rows = [r for r in rows if r.get("well_above_zero", True)]
    names = list(rows[0]["V_models"]) if rows else []
    res = {}
    for n in names:
        meas = np.array([r["plateau_V"] for r in rows])
        pred = np.array([r["V_models"][n] for r in rows])
        rel = (meas - pred) / pred
        res[n] = {"rmse_V": float(np.sqrt(np.mean((meas - pred) ** 2))),
                  "median_abs_rel_error": float(np.median(np.abs(rel))),
                  "rel_errors": rel.tolist()}
    return res


def wilcoxon_vs_full(rows, base="full model"):
    """Paired Wilcoxon signed-rank on |error|: full model vs each other model,
    same measurements. Returns statistic, p and n per comparison."""
    out = {}
    meas = np.array([r["plateau_V"] for r in rows])
    e_full = np.abs(meas - np.array([r["V_models"][base] for r in rows]))
    for n in rows[0]["V_models"]:
        if n == base:
            continue
        e = np.abs(meas - np.array([r["V_models"][n] for r in rows]))
        if len(e) < 6 or np.allclose(e, e_full):
            out[n] = {"n": len(e), "statistic": None, "p": None,
                      "note": "too few paired measurements" if len(e) < 6 else "identical"}
            continue
        w = stats.wilcoxon(e_full, e, alternative="less")
        out[n] = {"n": len(e), "statistic": float(w.statistic), "p": float(w.pvalue),
                  "full_model_better_in": int(np.sum(e_full < e))}
    return out


def decay_exponent(rows, n_boot=1000, seed=0, rmin_mm=None, rmax_mm=None):
    """Fit V ~ r^(-n) (log-log least squares) to the measured potentials;
    95% CI by bootstrap over sessions, then positions within each session."""
    rng = np.random.default_rng(seed)
    use = [r for r in rows if (rmin_mm is None or r["distance_mm"] >= rmin_mm)
           and (rmax_mm is None or r["distance_mm"] <= rmax_mm) and r["plateau_V"] > 0]
    by_s = {}
    for r in use:
        by_s.setdefault(str(r["session"]), []).append((r["distance_mm"], r["plateau_V"] / r["current_A"]))
    def fit(pairs):
        d, v = np.array(pairs).T
        return -np.polyfit(np.log(d), np.log(v), 1)[0]
    allp = [p for s in by_s.values() for p in s]
    n_hat = fit(allp)
    keys = list(by_s)
    boots = []
    for _ in range(n_boot):
        ks = rng.choice(keys, len(keys), replace=True)
        pairs = []
        for k in ks:
            ps = by_s[k]
            pairs += [ps[i] for i in rng.integers(0, len(ps), len(ps))]
        if len(set(p[0] for p in pairs)) >= 2:
            boots.append(fit(pairs))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"exponent": float(n_hat), "ci95": [float(lo), float(hi)], "n_sessions": len(keys),
            "n_measurements": len(allp),
            "note": ("only 2 sessions: the between-session spread is a range, not a real "
                     "confidence interval (plan 5.12)") if len(keys) <= 2 else ""}


def reciprocity(rows, floor_rel=0.02):
    """Swap the driven pair and the measured pair: V / I must not change, in any
    layout (no model input). Allowed difference, fixed in advance:
    sqrt(floor^2 + (2 sqrt2 noise / |V|)^2), with the noise floor from the
    optional noise_floor_V column."""
    out = []
    for r in rows:
        d = (r["V_swapped"] - r["V_forward"]) / r["V_forward"]
        nz = (r.get("noise_floor_V", 0) or 0)
        nz = float(nz) / abs(r["V_forward"]) if isinstance(nz, (int, float)) else 0.0
        tol = float(np.hypot(floor_rel, 2 * np.sqrt(2) * nz))
        out.append({"session": r["session"], "layout": r.get("layout", ""), "distance_mm": r["distance_mm"],
                    "rel_difference": float(d), "allowed": tol, "pass": bool(abs(d) <= tol)})
    return out


def linearity(rows):
    """V / I at each (session, position, pulse width) across amplitudes: should be constant."""
    groups = {}
    for r in rows:
        key = (str(r["session"]), r.get("layout", ""), str(r["position"]), r.get("condition", "A"),
               r.get("pulse_width_us"))
        groups.setdefault(key, []).append(r["plateau_V"] / r["current_A"])
    spreads = [np.ptp(v) / np.mean(v) for v in groups.values() if len(v) >= 2]
    return {"groups_with_2plus_amplitudes": len(spreads),
            "max_rel_spread_V_over_I": float(np.max(spreads)) if spreads else None}
