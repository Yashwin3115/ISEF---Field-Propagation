"""Pass criteria for the phantom experiments, FIXED BEFORE ANY MEASUREMENT.

The rules are written out for people in notes/phantom_pass_criteria.md; the
numbers they use are the constants in RULES below. runs/phantom_predict.py
resolves them for the measured geometry (which positions can tell the models
apart, how many readings must be covered) and writes pass_criteria.json next
to the predictions, so the criteria are committed under the same git tag.
runs/phantom_compare.py only APPLIES them: nothing here looks at data when
the criteria are set.

Two questions per experiment, answered separately:
  FULL MODEL PASSES?    enough readings inside the committed 95% band
  SIMPLE MODEL FAILS?   at the positions where the simple model (point source,
                        or a constant ratio) was predicted to differ from the
                        full model by more than the band, the data sit closer
                        to the full model
Four outcomes are possible and all are reported as they fall:
  pass + fails          the hypothesis is supported in this experiment
  pass + does not fail  the experiment could not tell the models apart
  not pass              the full model is rejected here; the result is
                        reported as a failed prediction. Any later refit is
                        labelled "fitted" and never replaces the blind result.
"""
import hashlib
import numpy as np
from scipy import stats

RULES = {
    "band_level": 0.95,
    "coverage_confidence": 0.95,       # k_min: P(covered count >= k_min | true coverage 0.95) >= this
    "min_discriminating": 3,           # fewer positions than this -> "not testable", said in advance
    "closer_fraction": 0.75,           # simple model fails if the full model is closer in >= this share
    "wilcoxon_alpha": 0.05,            # ... and (with >= 6 readings) the paired test agrees
    "reciprocity_floor_rel": 0.02,     # allowed forward/swapped difference before noise
    "flatness_floor_rel": 0.02,        # allowed spread of A - B across positions before noise
}


def k_min(n, p=None, conf=None):
    """Smallest number of covered readings (out of n) that is still consistent
    with a true 95% band: the largest k with P(X >= k | n, 0.95) >= 0.95."""
    p = RULES["band_level"] if p is None else p
    conf = RULES["coverage_confidence"] if conf is None else conf
    if n <= 0:
        return 0
    for k in range(n, -1, -1):
        if stats.binom.sf(k - 1, n, p) >= conf:
            return k
    return 0


def prior_meas_rel(unc):
    """95% relative measurement uncertainty expected from the stated inputs
    (conductivity meter, temperature, current, scope gain)."""
    u_sig = np.hypot(unc["sigma_meter_rel_sd"], unc["sigma_temp_coeff_per_C"] * unc["temperature_sd_C"])
    return float(2 * np.sqrt(u_sig ** 2 + unc["current_rel_sd"] ** 2 + unc["scope_gain_rel_sd"] ** 2))


def _disc_abs(rows, key, lo, hi, simple_key, m_rel):
    """Positions where the simple model lies outside the widened band."""
    out = []
    for k, r in enumerate(rows):
        G = r[key]
        wl = np.hypot(G - r[lo], m_rel * abs(G)); wh = np.hypot(r[hi] - G, m_rel * abs(G))
        s = r[simple_key]
        if s < G - wl or s > G + wh:
            out.append(k)
    return out


def resolve_criteria(experiments, derived, part7, setup, md_path=None):
    unc = setup["uncertainties"]
    m_rel = prior_meas_rel(unc)
    C = {"rules": RULES, "prior_measurement_uncertainty_95_rel": m_rel, "experiments": {}, "summary_lines": []}
    if md_path is not None and md_path.exists():
        C["rules_text_sha256"] = hashlib.sha256(md_path.read_bytes()).hexdigest()
        C["rules_text_file"] = "notes/phantom_pass_criteria.md"
    L = C["summary_lines"]
    L.append(f"expected measurement uncertainty (95%): {m_rel*100:.1f}% of each reading, plus its noise floor")
    for name, ex in experiments.items():
        rows = ex["rows"]
        d = [r["distance_mm"] for r in rows]
        e = {"n_positions": len(rows),
             "coverage_needed_per_condition": f"k_min(n) of n readings (n = {len(rows)} per session and "
                                              f"pulse setting -> {k_min(len(rows))})",
             "simple_models_condition_B": {}}
        for label, key in (("point, full space", "G_point_full_space_per_m"),
                           ("point, half space", "G_point_half_space_per_m"),
                           ("disk, half space", "G_disk_half_space_per_m"),
                           ("point, full space, referenced", "G_point_full_space_referenced_per_m"),
                           ("point, half space, referenced", "G_point_half_space_referenced_per_m")):
            if key not in rows[0]:
                continue
            idx = _disc_abs(rows, "G_B_per_m", "G_B_lo_per_m", "G_B_hi_per_m", key, m_rel)
            e["simple_models_condition_B"][label] = {
                "discriminating_positions": idx, "distances_mm": [d[k] for k in idx],
                "testable": len(idx) >= RULES["min_discriminating"]}
        C["experiments"][name] = e
        sm = e["simple_models_condition_B"]
        best = "point, full space, referenced" if ex.get("setup_part6", {}).get("support", {}).get("type") == "post" \
            else "point, half space, referenced"
        e["fairest_simple_model"] = best if best in sm else None
        L.append(f"{name}: full model needs >= {k_min(len(rows))}/{len(rows)} readings covered per series; "
                 "simple models differ from it at: " + "; ".join(
                     f"{k} {len(v['distances_mm'])} positions" for k, v in sm.items()))
        if best in sm:
            L.append(f"   fairest simple model here ({best}): distinguishable at "
                     f"{', '.join(f'{v:g}' for v in sm[best]['distances_mm']) or 'no'} mm")
    C["derived"] = {}
    def ratio_block(rows, simple):
        usable = [k for k, q in enumerate(rows) if q["usable"]]
        disc = [k for k in usable if not (rows[k]["lo"] / (1 + m_rel) <= simple <= rows[k]["hi"] * (1 + m_rel))]
        return {"usable_positions": usable, "simple_model_value": simple,
                "discriminating_positions": disc,
                "distances_mm": [rows[k]["distance_mm"] for k in disc],
                "coverage_needed": k_min(len(usable)), "testable": len(disc) >= RULES["min_discriminating"]}
    if "plate_ratio" in derived:
        b = ratio_block(derived["plate_ratio"]["condition_B"], 2.0)
        C["derived"]["plate_ratio"] = b
        L.append(f"plate ratio: {len(b['usable_positions'])} usable distances, >= {b['coverage_needed']} must be "
                 f"covered; 'exactly 2' is distinguishable at {len(b['distances_mm'])} "
                 f"({', '.join(f'{v:g}' for v in b['distances_mm'])} mm)")
    if "return_ratio" in derived:
        C["derived"]["return_ratio"] = {}
        for n, rows in derived["return_ratio"].items():
            if not isinstance(rows, list):
                continue
            b = ratio_block(rows, 1.0)
            C["derived"]["return_ratio"][n] = b
            L.append(f"return ratio {n}: {len(b['usable_positions'])} usable distances, >= {b['coverage_needed']} "
                     f"covered; 'no dependence on the return' is distinguishable at {len(b['distances_mm'])} "
                     f"({', '.join(f'{v:g}' for v in b['distances_mm'])} mm)")
    C["part7"] = []
    for rec in part7:
        gr = rec["gradient_ratio"]
        lo, hi = gr["band_with_assumed_noise"]
        C["part7"].append({
            "stim_offset_mm": rec["stim_offset_mm"],
            "uniform_medium_ratio": gr.get("uniform_medium_ratio"),
            "uniform_medium_outside_band_at_assumed_noise":
                bool(gr.get("uniform_medium_ratio") is not None and
                     not (lo <= gr["uniform_medium_ratio"] <= hi)),
            "rule": "full model passes if the measured ratio is inside its band (recomputed with the "
                    "session's noise); 'no interface' fails if its ratio is outside the measured ratio's "
                    "band"})
        L.append(f"Part 7, stim {rec['stim_offset_mm']}: predicted ratio {gr['fd_prediction']:.3f} "
                 f"[{lo:.3f}, {hi:.3f}] at the assumed noise; with no interface it would be "
                 f"{gr.get('uniform_medium_ratio', float('nan')):.3f} "
                 f"({'distinguishable' if C['part7'][-1]['uniform_medium_outside_band_at_assumed_noise'] else 'NOT distinguishable at this noise'})")
    return C


def closer_test(e_full, e_simple):
    """Is the full model closer to the data than the simple model? Returns the
    verdict by the committed rule and the numbers behind it."""
    e_full = np.asarray(e_full, float); e_simple = np.asarray(e_simple, float)
    n = len(e_full)
    if n == 0:
        return {"n": 0, "simple_model_fails": None, "note": "no readings at the discriminating positions"}
    frac = float(np.mean(e_full < e_simple))
    out = {"n": n, "full_closer_fraction": frac, "p": None}
    ok = frac >= RULES["closer_fraction"]
    if n >= 6 and not np.allclose(e_full, e_simple):
        w = stats.wilcoxon(e_full, e_simple, alternative="less")
        out["p"] = float(w.pvalue)
        ok = ok and w.pvalue < RULES["wilcoxon_alpha"]
    out["simple_model_fails"] = bool(ok)
    return out


def verdict(full_pass, simple_fails):
    if full_pass is None:
        return "no data"
    if not full_pass:
        return "FULL MODEL REJECTED here (reported as a failed blind prediction)"
    if simple_fails is None:
        return "full model passes; simple model not testable in this experiment"
    return ("SUPPORTED: full model passes and the simple model fails" if simple_fails else
            "full model passes, but the data do not separate it from the simple model")
