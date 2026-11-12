"""Phantom comparison (research plan v16, Parts 6-7 analysis): measured vs
COMMITTED predictions. Used by runs/phantom_compare.py; tested by
tests/test_phantom_compare.py on synthetic data.

Inputs (CSV, templates in notes/):
  measurements:  one row per (session, position, condition, pulse width, amplitude):
      part, session, position, condition, stim_index, pulse_width_us, current_A,
      current_rel_sd, plateau_V, plateau_V_sd, n_traces, noise_floor_V
      part 6: position = index into the committed distances (0-9);
              condition = A (scope ground at the return) or B (far reference wire)
      part 7: position = index along the committed traverse (0-11);
              condition = A
  sessions:      session, part, sigma_start, sigma_end, sigma_rel_sd,
                 temp_start_C, temp_end_C, sigma_in_start, sigma_in_end
                 (sigma_in_*: agar, part 7 only; sigma_* is the saline)
  reciprocity:   session, distance_mm, V_forward, V_swapped (optional)

Tests applied (all fixed before measurement, in predictions.json):
  coverage of every reading by its 95% band (conditions A and B, traverse)
  the A - B difference: predicted to be the SAME at every position
  the gradient ratio across the agar interface, from the same four points
  and differences as the prediction, corrected to the measured conductivities

Statistical units (plan 5.12): one row = one measurement (20 averaged traces
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
            "disk, half space": "G_disk_half_space_per_m"}


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


def predict_rows(meas, sessions, pred):
    """Attach V_pred (each applicable model) and the 95% band to every row."""
    sess = {str(s["session"]): s for s in sessions}
    out = []
    for m in meas:
        s = sess[str(m["session"])]
        sig, u_sig, drift = session_sigma(s)
        u_I = m.get("current_rel_sd", 0.0) or 0.0
        I = m["current_A"]
        r = dict(m)
        r["condition"] = _cond(m)
        if int(m["part"]) == 6:
            row = pred["part6"]["rows"][int(m["position"])]
            r["distance_mm"] = row["distance_mm"]
            if r["condition"] == "A":
                G, U_num, U_pos, U_ref = row["G_A_per_m"], row["U_num_A_rel"], row["U_pos_A_rel"], 0.0
                r["V_models"] = {"full model": I * G / sig}
            else:
                G, U_num, U_pos, U_ref = row["G_B_per_m"], row["U_num_B_rel"], row["U_pos_B_rel"], \
                    row.get("U_refpos_B_rel", 0.0)
                r["V_models"] = {name: I * row[k] / sig for name, k in MODELS_B.items()}
            r["V_pred"] = I * G / sig
            U = np.sqrt(U_num ** 2 + U_pos ** 2 + U_ref ** 2 + (2 * u_sig) ** 2 + (2 * u_I) ** 2)
        else:
            p7 = pred["part7"][int(m.get("stim_index", 0) or 0)]
            sig_in, u_in, _ = session_sigma(s, inner=True)
            ratio = sig_in / sig
            k = int(float(m["position"]))
            G0, U_num, dl, U_pos = p7["G_per_m"][k], p7["U_num_rel"][k], p7["dlnG_dln_ratio"][k], p7["U_pos_rel"][k]
            r["distance_mm"] = p7["traverse_distances_mm"][k]
            G = G0 * (ratio / p7["ratio_in_out_nominal"]) ** dl
            r["V_pred"] = I * G / sig
            r["V_models"] = {"full model": r["V_pred"]}
            u_ratio = np.hypot(u_sig, u_in) * abs(dl)
            U = np.sqrt(U_num ** 2 + U_pos ** 2 + (2 * u_sig) ** 2 + (2 * u_I) ** 2 + (2 * u_ratio) ** 2)
        r["U_rel"] = float(U)
        r["band_V"] = float(U * abs(r["V_pred"]) + 2 * (m.get("noise_floor_V", 0.0) or 0.0))
        r["covered"] = bool(abs(m["plateau_V"] - r["V_pred"]) <= r["band_V"])
        r["sigma_used"] = sig
        r["sigma_drift_rel"] = drift
        out.append(r)
    return out


def difference_test(rows, sessions, pred):
    """Part 6: V_A - V_B at each (session, position, pulse width, current)
    pair, against the committed position-independent difference."""
    sess = {str(s["session"]): s for s in sessions}
    D = pred["part6"]["difference_A_minus_B"]
    pairs = {}
    for r in rows:
        if int(r["part"]) != 6:
            continue
        key = (str(r["session"]), int(r["position"]), r.get("pulse_width_us"), round(r["current_A"], 9))
        pairs.setdefault(key, {})[r["condition"]] = r
    out = []
    for key, pr in pairs.items():
        if "A" not in pr or "B" not in pr:
            continue
        a, b = pr["A"], pr["B"]
        sig, u_sig, _ = session_sigma(sess[key[0]])
        I = 0.5 * (a["current_A"] + b["current_A"])
        u_I = max(a.get("current_rel_sd", 0) or 0, b.get("current_rel_sd", 0) or 0)
        Vp = I * D["G_A_minus_B_per_m"] / sig
        U = np.sqrt(D["U_num_rel"] ** 2 + D["U_refpos_rel"] ** 2 + (2 * u_sig) ** 2 + (2 * u_I) ** 2)
        noise = np.hypot(a.get("noise_floor_V", 0) or 0, b.get("noise_floor_V", 0) or 0)
        Vm = a["plateau_V"] - b["plateau_V"]
        out.append({"session": key[0], "position": key[1], "distance_mm": a["distance_mm"],
                    "pulse_width_us": key[2], "current_A": I, "V_meas": Vm, "V_pred": Vp,
                    "band_V": U * abs(Vp) + 2 * noise,
                    "covered": bool(abs(Vm - Vp) <= U * abs(Vp) + 2 * noise),
                    "V_meas_over_I": Vm / I})
    # flatness: is the measured difference the same at every position?
    flat = {}
    for sname in set(o["session"] for o in out):
        v = np.array([o["V_meas_over_I"] for o in out if o["session"] == sname])
        flat[sname] = {"n": len(v), "rel_spread": float(np.ptp(v) / abs(np.mean(v))) if len(v) > 1 else None}
    return out, flat


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
        U = np.sqrt(gr["U_num_rel"] ** 2 + gr["U_pos_rel"] ** 2 + U_noise ** 2 + (2 * u_ratio) ** 2)
        out.append({"session": key[0], "stim_index": key[1], "pulse_width_us": key[2],
                    "current_A": key[3], "R_meas": float(R_meas), "R_pred": float(R_pred),
                    "R_ideal_measured_sigmas": float(sig / sig_in),
                    "gap_fd_vs_ideal": float(R_pred / (sig / sig_in) - 1),
                    "U_rel": float(U), "U_noise_rel": U_noise,
                    "covered": bool(abs(R_meas - R_pred) <= U * abs(R_pred))})
    return out


def coverage(rows):
    c = np.array([r["covered"] for r in rows])
    return float(c.mean()) if len(c) else np.nan, int(c.sum()), len(c)


def errors_by_model(rows):
    """Relative error and RMSE per model (part 6 rows)."""
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


def reciprocity(rows):
    out = []
    for r in rows:
        d = (r["V_swapped"] - r["V_forward"]) / r["V_forward"]
        out.append({"session": r["session"], "distance_mm": r["distance_mm"], "rel_difference": float(d)})
    return out


def linearity(rows):
    """V / I at each (session, position, pulse width) across amplitudes: should be constant."""
    groups = {}
    for r in rows:
        key = (str(r["session"]), str(r["position"]), r.get("condition", "A"), r.get("pulse_width_us"))
        groups.setdefault(key, []).append(r["plateau_V"] / r["current_A"])
    spreads = [np.ptp(v) / np.mean(v) for v in groups.values() if len(v) >= 2]
    return {"groups_with_2plus_amplitudes": len(spreads),
            "max_rel_spread_V_over_I": float(np.max(spreads)) if spreads else None}
