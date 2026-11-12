"""Aim 2 as specified in research plan v13 (steps 19-30): calibration cost of
artifact removal.

Question: how many calibration pulses does each method need to recover spikes,
and does a physics template need fewer than an empirical one?

Synthetic ground truth (step 19; generated noise -- labelled as such):
  TRUE world: well-wall return, sigma 1.5 S/m, |Z| 450 kOhm (between the
  published 300-600), stimulator output resistance 10 kOhm, 500 mV biphasic,
  200 us per phase. Spikes from a library of 5 waveform classes
  (recovery.WAVEFORMS), 50-200 uV: background firing 20 Hz on every electrode,
  plus evoked spikes on 40% of electrodes (fixed latency 0.15-2.5 ms per
  electrode, jitter 30 us, fires on 60% of pulses). Noise 5 uV. Amplifier:
  +-3 mV range, 0.1 ms recovery, 20 kHz, 50 us averaging, and a first-order
  high-pass (AC coupling) either off or at 300 Hz -- the filter setting is
  undocumented, so both are run; the physics template applies the same
  (known, user-set) filter to i(t). With no filter the artifact is only the
  ~10 us current spikes at the pulse edges; the filter turns each into a
  millisecond tail, as seen in real recordings.
  Spike times, waveforms and amplitudes are kept in a separate array the
  methods never see.

Splits (step 20): 60 pulses -> calibration 30 / validation 10 / test 20,
  whole pulses, frozen in notes/aim2_split.json (created once with its seed;
  commit it; it is never regenerated). Templates come from calibration
  pulses; detector thresholds from validation pulses; the test split is
  scored once.

Methods (step 26), all with the same detector and the same budget:
  no correction
  blanking 0-1 ms with linear interpolation
  per-channel average template (mean of the budget's calibration pulses)
  low-rank empirical model: SVD of that mean, rank 1-20 swept, best rank
    chosen on the VALIDATION split at each budget (step 24)
  physics template, scale + time constant (step 22): (alpha Zhat) * i(t; tau)
  physics template + uniform offset: (alpha Zhat + beta) * i(t; tau)
    (the offset was found necessary on Nov 10; reported as an extension)
  The physics template is built from DELIBERATELY WRONG models (step 21):
  each return path x sigma +-10% x |Z| 300 / 600 kOhm = 12 variants.

Budgets (step 23): 1, 2, 5, 10, 20 calibration pulses, 10 random draws each.

Detector: negative threshold crossing at -k x noise (noise from the
  pre-pulse baseline, MAD). k is set per method, budget and draw on the
  validation split: the smallest k in 4.0-12.0 giving <= 2 false detections
  per electrode per second in 0-3 ms after the pulse ("fixed false-positive
  rate"). A detection matches a planted spike within +-2 samples (step 28).

Earliest recovery time (step 29): delay bins 0-0.25-0.5-0.75-1-1.5-2-2.5 ms;
  a bin passes if >= 90% of evoked spikes are found and <= 5% of detections
  in it are false; earliest recovery = start of the first bin from which every
  later bin passes. Non-saturated spikes only.

Effective degrees of freedom (step 25): physics = number of linear
  coefficients (trace of the hat matrix of the linear least squares) + 1
  nonlinear (tau); low-rank rank r on an n_el x n_t matrix = r (n_el + n_t - r);
  per-channel average = n_el x n_t.

Failure diagnostics (step 30) for the physics template at budget 20:
  spatial  -- per-electrode residual gain (residual projected on i(t))
              correlated with each return path's pattern
  waveform -- residual projected on i(t) and on d i / d tau
  clipping -- share of missed spikes that fall in saturated samples

Usage:
    python runs/aim2_budget.py            (~15-25 min)
    python runs/aim2_budget.py --quick    (2 draws, 3 budgets, 2 variants)
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
from maxone import transfer_run, PITCH, PROVISIONAL, um, mm
from interface import capacitance_from_impedance
from recovery import (WAVEFORMS, record, noise_level, detect_fast, fit_physics, lowrank,
                      blank_interp, unit_current_fine, bin_average, TS)

QUICK = "--quick" in sys.argv
label = "aim2_budget_quick" if QUICK else "aim2_budget"
OUT = ROOT / "results" / "runs" / label
FIG = ROOT / "figures" / "runs" / label
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
GRID = dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK else \
       dict(n_pad=6, growth=1.15, h_array=70 * um, h_far=0.7 * mm)

AMP, PHASE = 0.5, 200e-6
SIGMA0 = PROVISIONAL["sigma_S_per_m"]
TRUE = {"return": "well_wall", "sigma_factor": 1.0, "z1k": 450e3, "R_buf": 10e3}
VARIANTS = [(r, sf, z) for r in ["top_plate", "well_wall", "periphery"]
            for sf in (0.9, 1.1) for z in (300e3, 600e3)]
if QUICK:
    VARIANTS = [("top_plate", 1.1, 300e3), ("periphery", 0.9, 600e3)]
PROVISIONAL_VARIANT = ("top_plate", 1.1, 300e3)
BUDGETS = [1, 5, 20] if QUICK else [1, 2, 5, 10, 20]
N_DRAWS = 2 if QUICK else 10
RANGES_mV = [3]
HP_LIST = [300.0] if QUICK else [None, 300.0]          # amplifier high-pass corner (Hz); None = off
NOISE = 5e-6
RECOVER = 2
SUB = 20
T0, T1 = -1e-3, 5e-3
n = int(round((T1 - T0) / TS))
t_fine = T0 + (np.arange(n * SUB) + 0.5) * TS / SUB
t_samp = T0 + (np.arange(n) + 0.5) * TS
WIN = (t_samp > -0.1e-3) & (t_samp < 1.5e-3)          # where templates are fitted
POST = (t_samp >= 0) & (t_samp < 3e-3)                # where false detections are counted
BASE = t_samp < -0.05e-3
BINS = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
K_GRID = np.arange(4.0, 12.01, 0.5)
FP_TARGET = 2.0                                       # per electrode per second
C_GRID = np.geomspace(0.12e-9, 1.2e-9, 31)
rng = np.random.default_rng(20261111)
t_start = time.time()

# ------------------------------------------------------------------ fields
fields = {r: transfer_run({"return": r}, verbose=False, **GRID)
          for r in ["top_plate", "well_wall", "periphery"]}
ref = fields["top_plate"]
c0, r0 = PROVISIONAL["stim_col_row"]
near = np.flatnonzero((np.abs(ref["col"] - c0) <= 5) & (np.abs(ref["row"] - r0) <= 5)
                      & ~((ref["col"] == c0) & (ref["row"] == r0)))
row_sel = np.flatnonzero((ref["row"] == r0) & (ref["col"] != c0) & (np.abs(ref["col"] - c0) <= 120))
others = np.setdiff1d(np.flatnonzero(ref["distance_m"] > 0), np.union1d(near, row_sel))
el = np.unique(np.concatenate([near, row_sel, rng.choice(others, 100 if QUICK else 300, replace=False)]))
n_el = len(el)
dist_p = ref["distance_m"][el] / PITCH
print(f"fields solved ({time.time()-t_start:.0f} s); {n_el} electrodes, 1-{dist_p.max():.0f} pitches")

# ------------------------------------------------------------------ truth
Ft = fields[TRUE["return"]]
Z_true = Ft["transfer_ohm"][el] / TRUE["sigma_factor"]
R_true = Ft["R_solution_ohm"] / TRUE["sigma_factor"]
C_true = capacitance_from_impedance(TRUE["z1k"])
i_true = AMP * unit_current_fine(R_true, C_true, t_fine - T0, -T0, PHASE, TRUE["R_buf"])
art_fine = Z_true[:, None] * i_true[None, :]

N_PULSES = 60
split_file = ROOT / "notes" / ("aim2_split_quick.json" if QUICK else "aim2_split.json")
if split_file.exists():
    split = json.loads(split_file.read_text())
    print(f"split loaded from {split_file.relative_to(ROOT)} (frozen)")
else:
    seed = 777
    perm = np.random.default_rng(seed).permutation(N_PULSES).tolist()
    split = {"seed": seed, "n_pulses": N_PULSES, "calibration": sorted(perm[:30]),
             "validation": sorted(perm[30:40]), "test": sorted(perm[40:]),
             "created": time.strftime("%Y-%m-%d %H:%M"),
             "note": "frozen: commit this file; never regenerate (plan step 20)"}
    split_file.write_text(json.dumps(split, indent=2))
    print(f"split CREATED: {split_file.relative_to(ROOT)} -- commit it")

wf_names = list(WAVEFORMS)
wf_of = rng.integers(0, len(wf_names), n_el)            # each electrode's unit waveform
amp_of = rng.uniform(50e-6, 200e-6, n_el)
evoked_on = rng.random(n_el) < 0.4
lat_of = rng.uniform(0.15e-3, 2.5e-3, n_el)

def make_pulse(seed):
    r = np.random.default_rng(seed)
    sp = np.zeros_like(art_fine)
    truth = []                  # (el, sample, evoked?, latency_s, waveform)
    for k in range(n_el):
        times = []
        if evoked_on[k] and r.random() < 0.6:
            times.append((lat_of[k] + r.normal(0, 30e-6), True))
        nb = r.poisson(20 * (T1 - T0))
        times += [(tt, False) for tt in r.uniform(T0 + 0.5e-3, T1 - 1e-3, nb)]
        for tt, ev in times:
            sp[k] += amp_of[k] * WAVEFORMS[wf_names[wf_of[k]]]((t_fine - tt) * 1e3)
            truth.append((k, int(np.floor((tt - T0) / TS)), ev, tt, wf_of[k]))
    return sp, truth

pulses = {}
for p in range(N_PULSES):
    pulses[p] = make_pulse(1000 + p)
print(f"{N_PULSES} pulses generated ({time.time()-t_start:.0f} s)")

# ------------------------------------------------------------------ scoring
def score(clean, usable, sigma, k, truth, tol=2):
    """Vectorized scoring: found mask over truth spikes; false detections
    (event times) not within +-tol samples of any planted spike."""
    e, s = detect_fast(clean, usable, sigma, k)
    hit = np.zeros(clean.shape, bool); hit[e, s] = True
    near_spike = np.zeros(clean.shape, bool)
    tr = np.array([(a, b) for a, b, *_ in truth], int).reshape(-1, 2)
    for d in range(-tol, tol + 1):
        ss = np.clip(tr[:, 1] + d, 0, clean.shape[1] - 1)
        near_spike[tr[:, 0], ss] = True
    cs = np.concatenate([np.zeros((clean.shape[0], 1), int), np.cumsum(hit, 1)], 1)
    lo = np.clip(tr[:, 1] - tol, 0, clean.shape[1]); hi = np.clip(tr[:, 1] + tol + 1, 0, clean.shape[1])
    found = (cs[tr[:, 0], hi] - cs[tr[:, 0], lo]) > 0
    fp = ~near_spike[e, s]
    return found, s[fp]

def fp_rate(fp_samples, n_pulse):
    m = POST[fp_samples] if len(fp_samples) else np.zeros(0, bool)
    return m.sum() / (n_el * n_pulse * (POST.sum() * TS))

def calibrate_k(clean_list, usable_list, sigma_list, truth_list, _inst=None):
    """Smallest k with validation false-detection rate <= FP_TARGET."""
    for k in K_GRID:
        fps = [score(c, u, s, k, tr)[1] for c, u, s, tr in zip(clean_list, usable_list, sigma_list, truth_list)]
        if fp_rate(np.concatenate(fps), len(clean_list)) <= FP_TARGET:
            return float(k)
    return float(K_GRID[-1])

def evaluate(clean_list, usable_list, sigma_list, truth_list, inst_list, k):
    """Test-split statistics at threshold k. A spike is 'saturated' (excluded)
    only if the INSTRUMENT could not record its trough; samples a method
    discards itself (blanking) still count as misses."""
    found_all, ev, lat, wf, sat, dist_all, fp_all = [], [], [], [], [], [], []
    for c, u, s, tr, ui in zip(clean_list, usable_list, sigma_list, truth_list, inst_list):
        f, fp = score(c, u, s, k, tr)
        found_all.append(f)
        ev.append(np.array([x[2] for x in tr], bool))
        lat.append(np.array([x[3] for x in tr]))
        wf.append(np.array([x[4] for x in tr]))
        sat.append(np.array([not ui[x[0], min(max(x[1], 0), n - 1)] for x in tr], bool))
        dist_all.append(dist_p[np.array([x[0] for x in tr], int)])
        fp_all.append(fp)
    f, ev, lat, wf, sat, dd = map(np.concatenate, (found_all, ev, lat, wf, sat, dist_all))
    fps = np.concatenate(fp_all)
    res = {"k": k, "fp_rate": float(fp_rate(fps, len(clean_list))), "bins": {}}
    ok_after = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = ev & ~sat & (lat >= a * 1e-3) & (lat < b * 1e-3)
        tp = int(f[m].sum())
        fpb = int(((t_samp[fps] >= a * 1e-3) & (t_samp[fps] < b * 1e-3)).sum())
        rec = tp / max(m.sum(), 1)
        fdr = fpb / max(tp + fpb, 1)
        res["bins"][f"{a}-{b}"] = {"recall": rec, "n": int(m.sum()), "false_fraction": fdr,
                                   "passes": bool(rec >= 0.9 and fdr <= 0.05)}
        ok_after.append(res["bins"][f"{a}-{b}"]["passes"])
    earliest = None
    for i in range(len(ok_after)):
        if all(ok_after[i:]):
            earliest = BINS[i]; break
    res["earliest_recovery_ms"] = earliest
    m_ev = ev & ~sat
    res["recall_evoked_0_1ms"] = float(f[m_ev & (lat < 1e-3)].mean()) if (m_ev & (lat < 1e-3)).any() else None
    res["recall_evoked_all"] = float(f[m_ev].mean())
    res["recall_by_waveform"] = {wf_names[w]: float(f[m_ev & (wf == w) & (lat < 1e-3)].mean())
                                 for w in range(len(wf_names)) if (m_ev & (wf == w) & (lat < 1e-3)).any()}
    res["evoked_saturated_fraction"] = float(sat[ev].mean())
    return res, f, ev, lat, sat

# ------------------------------------------------------------------ diagnostics
from recovery import unit_current_binned

def _diagnose(tmpl, C, Rh, Zh, test_ids, data, found, ev, sat, r, blank_earliest):
    """Step 30: attribute a physics-template failure to space, time or clipping."""
    Rm = np.mean([data[p][0] - tmpl for p in test_ids], axis=0)
    Um = np.all([data[p][1] for p in test_ids], axis=0) & WIN[None, :]
    ib = unit_current_binned(Rh, C, t_fine - T0, -T0, SUB, PHASE, HP)
    dib = (unit_current_binned(Rh, C * 1.05, t_fine - T0, -T0, SUB, PHASE, HP) - ib) / np.log(1.05)
    A = np.zeros(n_el); B = np.zeros(n_el)
    for k in range(n_el):
        m = Um[k]
        if m.sum() < 3:
            continue
        X = np.column_stack([ib[m], dib[m]])
        A[k], B[k] = np.linalg.lstsq(X, Rm[k, m], rcond=None)[0]
    e_space = float(np.sum(A ** 2) * np.sum(ib[WIN] ** 2))
    e_time = float(np.sum(B ** 2) * np.sum(dib[WIN] ** 2))
    # which candidate return path explains the residual: the part of each
    # candidate's pattern that the template could NOT already represent
    # (orthogonal to its spatial basis Zhat and a constant) is compared with
    # the per-electrode residual gains A; R^2 of that one-pattern fit.
    B0 = np.column_stack([Zh, np.ones(n_el)])
    proj = lambda v: v - B0 @ np.linalg.lstsq(B0, v, rcond=None)[0]
    A_perp = proj(A)
    corr = {}
    for ret in fields:
        P = proj(fields[ret]["transfer_ohm"][el])
        nP = np.dot(P, P)
        corr[ret] = float(np.dot(A_perp, P) ** 2 / (nP * np.dot(A_perp, A_perp))) \
            if nP > 1e-12 * np.dot(fields[ret]["transfer_ohm"][el], fields[ret]["transfer_ohm"][el]) else 0.0
    # misses that count against the method are non-saturated ones (saturated
    # spikes are already excluded from recall). Clipping can still be the
    # cause through its AFTERMATH (the filter's memory after the recovery
    # window): share of those misses on electrodes that clipped in that trial.
    missed = ev & ~found & ~sat
    clipped_el = np.zeros(n_el, bool)
    for p_ in test_ids:
        clipped_el |= ~data[p_][1].all(axis=1)
    miss_el = np.concatenate([np.array([x[0] for x in data[p_][3]], int) for p_ in test_ids])
    clip_share = float(clipped_el[miss_el[missed]].mean()) if missed.any() else 0.0
    fails = r["earliest_recovery_ms"] is None or (blank_earliest is not None and
                                                  r["earliest_recovery_ms"] >= blank_earliest)
    if not fails:
        attribution = "meets criteria"
    elif clip_share > 0.5:
        attribution = "clipping aftermath (misses on channels that clipped)"
    elif e_space >= e_time:
        attribution = f"spatial pattern (residual best explained by the {max(corr, key=corr.get)} return)"
    else:
        attribution = "waveform (time constant)"
    return {"fails": bool(fails), "attribution": attribution, "residual_energy_spatial": e_space,
            "residual_energy_waveform": e_time, "residual_R2_by_return_pattern": corr,
            "missed_on_clipped_channels_share": clip_share}


# ------------------------------------------------------------------ main loop
results = {"settings": {"true_world": TRUE, "budgets": BUDGETS, "draws": N_DRAWS, "variants": VARIANTS,
                        "n_electrodes": n_el, "noise_uV": NOISE * 1e6, "fp_target_per_el_s": FP_TARGET,
                        "noise": "generated (white) -- replace with phantom noise when recorded",
                        "waveforms": "parametric classes (recovery.WAVEFORMS), not digitized published templates"},
           "split": split, "by_range": {}}
import itertools
for rng_mV, HP in itertools.product(RANGES_mV, HP_LIST):
    sat_V = rng_mV * 1e-3
    print(f"\n===== amplifier range +-{rng_mV} mV, high-pass {HP or 'off'} Hz =====")
    rec_rng = np.random.default_rng(5)
    data = {}
    for p in range(N_PULSES):
        sp, truth = pulses[p]
        # after clipping, the high-pass carries a wrong tail for a few of its
        # time constants: the clipped channel is unusable until it settles (3 tau)
        rec_n = RECOVER if not HP else int(np.ceil(3 / (2 * np.pi * HP) / TS))
        y, s = record(art_fine, sp, NOISE, sat_V, SUB, rec_n, rec_rng, hp_hz=HP)
        data[p] = (y, ~s, noise_level(y, BASE), truth)
    V, T = split["validation"], split["test"]
    def lists(ids, clean_fn):
        """(cleaned, usable, noise, truth, instrument-usable) for the pulses
        `ids`; methods may shrink `usable` (blanking), but SATURATION for
        scoring always comes from the instrument mask (the last entry)."""
        return ([clean_fn(data[p][0], p) for p in ids], [data[p][1] for p in ids],
                [data[p][2] for p in ids], [data[p][3] for p in ids], [data[p][1] for p in ids])

    out = {"methods": {}, "physics_variants": {}, "diagnostics": {}}
    # --- methods that need no calibration
    for name, fn, umask in [("no correction", lambda y, p: y, None),
                            ("blanking 1 ms + interpolation", lambda y, p: blank_interp(y, t_samp, 0, 1e-3)[0],
                             (t_samp >= 0) & (t_samp < 1e-3))]:
        cv, uv, sv, tv, iv = lists(V, fn)
        if umask is not None:
            uv = [u & ~umask[None, :] for u in uv]
        k = calibrate_k(cv, uv, sv, tv)
        ct, ut, st, tt, it = lists(T, fn)
        if umask is not None:
            ut = [u & ~umask[None, :] for u in ut]
        r, *_ = evaluate(ct, ut, st, tt, it, k)
        out["methods"][name] = {str(b): [r] for b in BUDGETS}       # budget-independent
        print(f"{name:34s} k {k:4.1f}  recall(evoked <1 ms) {r['recall_evoked_0_1ms']}  "
              f"earliest {r['earliest_recovery_ms']} ms  FP {r['fp_rate']:.2f}/el/s")

    # --- calibrated methods, budget x draws
    draw_rng = np.random.default_rng(99)
    draws = {b: [sorted(draw_rng.choice(split["calibration"], b, replace=False).tolist())
                 for _ in range(N_DRAWS)] for b in BUDGETS}
    results["draws"] = draws
    n_t = n
    for b in BUDGETS:
        for d_i, cal in enumerate(draws[b]):
            M = np.mean([data[p][0] for p in cal], axis=0)
            U_all = np.all([data[p][1] for p in cal], axis=0)
            # per-channel average template
            tmpl = M.copy()
            cv, uv, sv, tv, iv = lists(V, lambda y, p: y - tmpl)
            k = calibrate_k(cv, uv, sv, tv)
            r, *_ = evaluate(*lists(T, lambda y, p: y - tmpl), k)
            r["dof"] = n_el * n_t; r["nominal_params"] = n_el * n_t
            out["methods"].setdefault("per-channel average", {}).setdefault(str(b), []).append(r)
            # low-rank: rank sweep on validation
            best = None
            for rk in range(1, 21):
                Mr = lowrank(np.where(U_all, M, 0.0), rk)
                cv, uv, sv, tv, iv = lists(V, lambda y, p: y - Mr)
                kk = calibrate_k(cv, uv, sv, tv)
                rv, *_ = evaluate(cv, uv, sv, tv, iv, kk)
                score_v = rv["recall_evoked_0_1ms"] or 0.0
                if best is None or score_v > best[0]:
                    best = (score_v, rk, kk, Mr)
            _, rk, kk, Mr = best
            r, *_ = evaluate(*lists(T, lambda y, p: y - Mr), kk)
            r["rank"] = rk; r["dof"] = rk * (n_el + n_t - rk); r["nominal_params"] = rk * (n_el + n_t - rk)
            out["methods"].setdefault("low-rank empirical (best rank)", {}).setdefault(str(b), []).append(r)
            # physics variants: fitted to the mean of the calibration pulses
            # (the artifact is the same every pulse; averaging only lowers noise)
            for var in VARIANTS:
                ret, sf, z1k = var
                Zh = fields[ret]["transfer_ohm"][el] / sf
                Rh = fields[ret]["R_solution_ohm"] / sf
                for arm, off in [("physics (scale, tau)", False), ("physics + offset", True)]:
                    model, coef, C = fit_physics(M, U_all, Zh, Rh, C_GRID, t_fine - T0, -T0, SUB,
                                                 PHASE, window=WIN, offset=off, hp_hz=HP)
                    tmpl = model
                    cv, uv, sv, tv, iv = lists(V, lambda y, p: y - tmpl)
                    k = calibrate_k(cv, uv, sv, tv)
                    r, f, ev, lat, sat = evaluate(*lists(T, lambda y, p: y - tmpl), k)
                    r["C_fit_nF"] = C * 1e9; r["coef"] = list(map(float, coef))
                    r["dof"] = (2 if off else 1) + 1; r["nominal_params"] = (2 if off else 1) + 1
                    key = f"{arm} | {ret}, sigma x{sf}, {z1k/1e3:.0f} kOhm"
                    out["physics_variants"].setdefault(key, {}).setdefault(str(b), []).append(r)
                    if var == PROVISIONAL_VARIANT:
                        out["methods"].setdefault(arm, {}).setdefault(str(b), []).append(r)
                    if b == BUDGETS[-1] and d_i == 0:
                        be = out["methods"]["blanking 1 ms + interpolation"][str(b)][0]["earliest_recovery_ms"]
                        out["diagnostics"][key] = _diagnose(tmpl, C, Rh, Zh, T, data, f, ev, sat, r, be)
        print(f"budget {b:2d}: done ({time.time()-t_start:.0f} s)")
    results["by_range"][f"{rng_mV} mV, HP {HP or 'off'}"] = out

# ------------------------------------------------------------------ summary
def med_iqr(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None, None, None
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))

ORDER = ["no correction", "blanking 1 ms + interpolation", "per-channel average",
         "low-rank empirical (best rank)", "physics (scale, tau)", "physics + offset"]
COLORS = {"no correction": "0.6", "blanking 1 ms + interpolation": "#eb6834",
          "per-channel average": "#1baf7a", "low-rank empirical (best rank)": "#7b4fd0",
          "physics (scale, tau)": "#2a78d6", "physics + offset": "#111111"}
summary = {"settings": results["settings"], "split": split, "by_range": {}}
for rng_key, out in results["by_range"].items():
    print(f"\n===== {rng_key}: recall of evoked spikes < 1 ms after the pulse, at "
          f"<= {FP_TARGET:g} false/el/s (median [IQR] over {N_DRAWS} draws); physics = provisional "
          f"model {PROVISIONAL_VARIANT}")
    print(f"{'method':32s} " + " ".join(f"{'b=' + str(b):>17s}" for b in BUDGETS))
    S = {}
    for m in ORDER:
        row, S[m] = [], {}
        for b in BUDGETS:
            rs = out["methods"][m][str(b)]
            md, lo, hi = med_iqr([r["recall_evoked_0_1ms"] for r in rs])
            ea = med_iqr([r["earliest_recovery_ms"] for r in rs])[0]
            S[m][str(b)] = {"recall_median": md, "recall_iqr": [lo, hi], "earliest_median_ms": ea,
                            "fraction_draws_recall_ge_0.9": float(np.mean([(r["recall_evoked_0_1ms"] or 0) >= 0.9 for r in rs])),
                            "dof_median": med_iqr([r.get("dof", 0) for r in rs])[0],
                            "rank_median": med_iqr([r.get("rank") for r in rs])[0] if m.startswith("low") else None}
            row.append(f"{md:.2f} [{lo:.2f},{hi:.2f}]" if md is not None else "-")
        first = next((b for b in BUDGETS if (S[m][str(b)]["recall_median"] or 0) >= 0.9), None)
        S[m]["first_budget_median_recall_ge_0.9"] = first
        print(f"{m:32s} " + " ".join(f"{c:>17s}" for c in row) + f"   reaches 0.9 at b = {first}")
    print("earliest recovery time (median ms over draws):")
    for m in ORDER:
        print(f"  {m:32s} " + " ".join(f"{str(S[m][str(b)]['earliest_median_ms']):>6s}" for b in BUDGETS))
    print("effective degrees of freedom (median):  " + "; ".join(
        f"{m}: {S[m][str(BUDGETS[-1])]['dof_median']:.0f}" for m in ORDER[2:]))
    print(f"physics variants (model error stress test), recall < 1 ms at b = {BUDGETS[0]} and {BUDGETS[-1]}:")
    PV = {}
    for key, byb in out["physics_variants"].items():
        a = med_iqr([r["recall_evoked_0_1ms"] for r in byb[str(BUDGETS[0])]])[0]
        z = med_iqr([r["recall_evoked_0_1ms"] for r in byb[str(BUDGETS[-1])]])[0]
        dg = out["diagnostics"].get(key, {})
        PV[key] = {"recall_b_min": a, "recall_b_max": z, "diagnosis": dg}
        print(f"  {key:62s} {a:.2f} -> {z:.2f}   {dg.get('attribution', '')}")
    counts = {}
    for dg in out["diagnostics"].values():
        counts[dg["attribution"]] = counts.get(dg["attribution"], 0) + 1
    print("failure attribution counts:", counts)
    summary["by_range"][rng_key] = {"methods": S, "physics_variants": PV, "attribution_counts": counts,
                                    "no_calibration_methods": {m: out["methods"][m][str(BUDGETS[0])][0]
                                                               for m in ORDER[:2]}}
    # figure: recall vs budget
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
    for m in ORDER:
        md = [S[m][str(b)]["recall_median"] for b in BUDGETS]
        lo = [S[m][str(b)]["recall_iqr"][0] for b in BUDGETS]
        hi = [S[m][str(b)]["recall_iqr"][1] for b in BUDGETS]
        axs[0].plot(BUDGETS, [np.nan if v is None else v for v in md], "o-", color=COLORS[m], lw=2, label=m)
        if None not in lo + hi:
            axs[0].fill_between(BUDGETS, lo, hi, color=COLORS[m], alpha=0.12, lw=0)
        ea = [S[m][str(b)]["earliest_median_ms"] for b in BUDGETS]
        axs[1].plot(BUDGETS, [np.nan if e is None else e for e in ea], "o-", color=COLORS[m], lw=2)
    axs[0].axhline(0.9, color="0.5", lw=1, ls="--")
    axs[0].set_xscale("log"); axs[0].set_xticks(BUDGETS); axs[0].set_xticklabels(BUDGETS)
    axs[0].set_xlabel("calibration pulses"); axs[0].set_ylabel("evoked spikes found, < 1 ms after pulse")
    axs[0].set_title(f"recall at <= {FP_TARGET:g} false detections /el/s (median, IQR over draws)", fontsize=9.5)
    axs[0].legend(frameon=False, fontsize=8); axs[0].set_ylim(-0.02, 1.02)
    axs[1].set_xscale("log"); axs[1].set_xticks(BUDGETS); axs[1].set_xticklabels(BUDGETS)
    axs[1].set_xlabel("calibration pulses"); axs[1].set_ylabel("earliest recovery time (ms)")
    axs[1].set_title("earliest delay bin from which every bin passes (90% found, <= 5% false)", fontsize=9.5)
    for ax in axs:
        ax.grid(True, color="0.92", lw=0.5)
    fig.suptitle(f"Aim 2 calibration budget, amplifier +-{rng_key}; synthetic (generated noise)"
                 + (" -- QUICK" if QUICK else ""))
    fig.savefig(FIG / f"budget_{rng_key.replace(' ', '').replace(',', '_')}.png", dpi=200)
    plt.close(fig)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
(OUT / "full_results.json").write_text(json.dumps(results, default=float))
print(f"\ntotal {time.time()-t_start:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
