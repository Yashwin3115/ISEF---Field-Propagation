"""Aim 2 as specified in research plan v16 (steps 19-31): calibration cost of
artifact removal, robustness to pulse-to-pulse variability.

Question: how many calibration pulses does each method need to recover spikes,
does a physics template need fewer than an empirical one, and does that hold
when the artifact varies from pulse to pulse?

Synthetic ground truth (step 19; GENERATED noise -- labelled as such):
  TRUE world: well-wall return, sigma 1.5 S/m, |Z| 450 kOhm (between the
  published 300-600), stimulator output resistance 10 kOhm, 500 mV biphasic,
  200 us per phase. Spikes from 5 PARAMETRIC waveform classes
  (recovery.WAVEFORMS; not digitized published templates), 50-200 uV:
  background firing 20 Hz on every electrode, plus evoked spikes on 40% of
  electrodes (fixed latency 0.15-2.5 ms per electrode, jitter 30 us, fires on
  60% of pulses). Noise 5 uV. Amplifier: +-3 mV, 20 kHz, 50 us averaging,
  first-order high-pass off or at 300 Hz (setting undocumented; the 300 Hz
  corner is a stated basis: Ronchi et al. 2019 split their noise figures at
  300 Hz -- not confirmation of what the lab runs). After clipping, the channel
  is unusable for 0.1 ms (no filter) or 3 filter time constants (with filter).
  Spike times, waveforms and amplitudes stay in a separate array.

Pulse-to-pulse VARIABILITY (step 27), levels swept from zero:
  per pulse: amplitude x N(1, a); interface capacitance (so tau) x N(1, c);
  slow baseline drift: a linear trend across the pulse train, d x g_k at
  each electrode (g_k ~ N(0, 1)), from -d to +d over the 60 pulses.
  Levels (ASSUMED -- no published pulse-to-pulse figures were found):
    none (0, 0, 0), low (1%, 2%, 5 uV), medium (3%, 5%, 10 uV), high (10%, 10%, 25 uV)

Splits (step 20): 60 pulses -> calibration 30 / validation 10 / test 20,
  whole pulses, frozen in notes/aim2_split.json. Templates from calibration
  pulses only; detector thresholds from validation pulses; test scored once.

Processing arms (step 26), same detector, same budget, same traces:
  1 no correction
  2 blanking 0-1 ms with linear interpolation
  3 per-channel average template (mean of the budget's calibration pulses,
    each sample averaged over the pulses in which it was usable)
  4 low-rank empirical, FIXED: SVD of that mean, rank 1-20 swept, best rank
    chosen on the validation split at each budget (step 24)
  5 low-rank empirical, PER-PULSE: the spatial basis (rank r, chosen the same
    way) frozen from calibration; its coefficients refitted on each pulse
    from that pulse's own trace
  6 physics template, FIXED: (alpha Zhat + beta) i(t; tau), the three numbers
    fitted on the calibration pulses and held fixed (step 22)
  7 physics template, PER-PULSE: the same three numbers refitted on each pulse
    from that pulse's own trace (no calibration pulses needed -> flat line)
  supplementary: physics fixed with scale + tau only (no offset), so the
    offset's contribution stays visible (step 22)
  8 SALPA (step 28): the ORIGINAL implementation (author's PyPI package
    `salpa` 1.0, GPL-3.0-or-later, installed not copied) with the author's
    command-line defaults; per channel, no calibration -> flat line. Its main
    comparison is physics per-pulse. Reported twice: the author's pure
    defaults, and with the original tool's forced peg over the stimulus
    itself (0-0.45 ms; set by the pulse, not tuned). Skipped if not installed.
  The physics template is built from the provisional WRONG model (top-plate
  return, sigma x1.1, 300 kOhm); at variability "none" the full stress test
  of step 21 (each return path x sigma +-10% x 300 / 600 kOhm) is also run.

Spike-absorption check (step 22): the per-pulse physics fit is repeated on the
  same test pulses with the spikes removed (same noise); the shift in each of
  the three parameters is reported against that parameter's pulse-to-pulse
  spread.

Budgets (step 23): 1, 2, 5, 10, 20 calibration pulses, 10 random draws each.
Detector: threshold -k x noise, k set per arm/budget/draw on the validation
  split as the smallest k in 4-12 giving <= 2 false detections per electrode
  per second in 0-3 ms (fixed false-positive rate). Match tolerance +-2
  samples (step 29). Earliest recovery time (step 30): delay bins 0-0.25-0.5-
  0.75-1-1.5-2-2.5 ms; a bin passes with >= 90% found and <= 5% false; earliest
  = start of the first bin from which every later bin passes.
Effective degrees of freedom (step 25): physics = linear coefficients (trace of
  the hat matrix) + 1 nonlinear; low-rank fixed r (n_el + n_t - r); low-rank
  per-pulse r x n_t per pulse (+ the frozen basis); per-channel n_el x n_t.
Failure diagnostics (step 31) for the physics template at budget 20.

Usage:
    python runs/aim2_budget.py            (~1.5-3 h; leave it running)
    python runs/aim2_budget.py --quick    (2 draws, 3 budgets, 2 levels, ~3 min)
"""
import sys
import json
import time
import itertools
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
                      blank_interp, unit_current_fine, unit_current_binned, bin_average, TS,
                      salpa_available, salpa_clean, SALPA_LABEL, SALPA_PEG_LABEL, SALPA_DEFAULTS)

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
LEVELS = {"none": (0.0, 0.0, 0.0), "low": (0.01, 0.02, 5e-6),
          "medium": (0.03, 0.05, 10e-6), "high": (0.10, 0.10, 25e-6)}
if QUICK:
    LEVELS = {k: LEVELS[k] for k in ("none", "high")}
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
drift_pattern = np.random.default_rng(4242).normal(0, 1, n_el)   # g_k, fixed per electrode

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
    """Smallest k in K_GRID with validation false-detection rate <= FP_TARGET,
    found by bisection (the false-detection rate falls as k rises)."""
    def ok(k):
        fps = [score(c, u, s, k, tr)[1] for c, u, s, tr in zip(clean_list, usable_list, sigma_list, truth_list)]
        return fp_rate(np.concatenate(fps), len(clean_list)) <= FP_TARGET
    lo, hi = 0, len(K_GRID) - 1
    if ok(K_GRID[lo]):
        return float(K_GRID[lo])
    if not ok(K_GRID[hi]):
        return float(K_GRID[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(K_GRID[mid]):
            hi = mid
        else:
            lo = mid
    return float(K_GRID[hi])

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
    """Step 31: attribute a physics-template failure to space, time or clipping."""
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


# ------------------------------------------------------------------ helpers
def lowrank_basis(M, U_all, r):
    U, s, Vt = np.linalg.svd(np.where(U_all, M, 0.0), full_matrices=False)
    return U[:, :r]

def perpulse_project(y, u, Ur):
    """Artifact estimate for ONE pulse: at each sample, least-squares fit of
    the frozen spatial basis Ur to the usable electrodes of this pulse."""
    out = Ur @ (Ur.T @ np.where(u, y, 0.0))
    partial = np.flatnonzero(~u.all(axis=0))
    for t_ in partial:
        m = u[:, t_]
        if m.sum() <= Ur.shape[1]:
            out[:, t_] = 0.0
            continue
        c, *_ = np.linalg.lstsq(Ur[m], y[m, t_], rcond=None)
        out[:, t_] = Ur @ c
    return out

def physics_perpulse(y, u, Zh, Rh, HP):
    model, coef, C = fit_physics(y, u, Zh, Rh, C_GRID, t_fine - T0, -T0, SUB, PHASE,
                                 window=WIN, offset=True, hp_hz=HP, refine=True)
    return model, coef, C

def make_data(HP, level, with_spikes=True, ids=None):
    """Recorded traces for every pulse under filter HP and variability level."""
    a_j, c_j, d_v = LEVELS[level]
    jr = np.random.default_rng(31 + list(LEVELS).index(level))
    amp_f = 1 + a_j * jr.normal(size=N_PULSES)
    cap_f = np.clip(1 + c_j * jr.normal(size=N_PULSES), 0.5, 1.5)
    rec_n = RECOVER if not HP else int(np.ceil(3 / (2 * np.pi * HP) / TS))
    data, truth_par = {}, {}
    for p in (range(N_PULSES) if ids is None else ids):
        i_p = AMP * amp_f[p] * unit_current_fine(R_true, C_true * cap_f[p], t_fine - T0, -T0, PHASE,
                                                 TRUE["R_buf"])
        drift = d_v * (2 * p / (N_PULSES - 1) - 1) * drift_pattern
        art = Z_true[:, None] * i_p[None, :] + drift[:, None]
        sp, truth = pulses[p]
        y, s = record(art, sp if with_spikes else np.zeros_like(sp), NOISE, SAT_V, SUB, rec_n,
                      np.random.default_rng(5000 + p), hp_hz=HP)
        data[p] = (y, ~s, noise_level(y, BASE), truth)
        truth_par[p] = {"amplitude_factor": float(amp_f[p]), "capacitance_factor": float(cap_f[p])}
    return data, truth_par


# ------------------------------------------------------------------ main loop
SAT_V = RANGES_mV[0] * 1e-3
results = {"settings": {"true_world": TRUE, "budgets": BUDGETS, "draws": N_DRAWS, "variants": VARIANTS,
                        "variability_levels": {k: {"amplitude_jitter": v[0], "tau_jitter": v[1],
                                                   "drift_V": v[2]} for k, v in LEVELS.items()},
                        "variability_note": "ASSUMED levels; no published pulse-to-pulse figures found",
                        "n_electrodes": n_el, "noise_uV": NOISE * 1e6, "fp_target_per_el_s": FP_TARGET,
                        "noise": "generated (white) -- replace with phantom noise when recorded",
                        "waveforms": "parametric classes (recovery.WAVEFORMS), not digitized published templates",
                        "high_pass_basis": "300 Hz: Ronchi et al. 2019 report noise split at 300 Hz "
                                           "(stated basis, not confirmation of the lab's setting)"},
           "split": split, "by_condition": {}}
draw_rng = np.random.default_rng(99)
draws = {b: [sorted(draw_rng.choice(split["calibration"], b, replace=False).tolist())
             for _ in range(N_DRAWS)] for b in BUDGETS}
results["draws"] = draws
V, T = split["validation"], split["test"]
n_t = n
Zp = fields[PROVISIONAL_VARIANT[0]]["transfer_ohm"][el] / PROVISIONAL_VARIANT[1]
Rp = fields[PROVISIONAL_VARIANT[0]]["R_solution_ohm"] / PROVISIONAL_VARIANT[1]

for HP, level in itertools.product(HP_LIST, LEVELS):
    cond = f"HP {HP or 'off'} | variability {level}"
    print(f"\n===== amplifier +-{RANGES_mV[0]} mV, high-pass {HP or 'off'} Hz, variability {level} =====",
          flush=True)
    data, truth_par = make_data(HP, level)
    def lists(ids, clean_fn):
        return ([clean_fn(data[p][0], p) for p in ids], [data[p][1] for p in ids],
                [data[p][2] for p in ids], [data[p][3] for p in ids], [data[p][1] for p in ids])
    out = {"methods": {}, "physics_variants": {}, "diagnostics": {}, "truth_parameters": truth_par}

    def run_flat(name, fn, umask=None, extra=None):
        cv, uv, sv, tv, iv = lists(V, fn)
        if umask is not None:
            uv = [u & ~umask[None, :] for u in uv]
        k = calibrate_k(cv, uv, sv, tv)
        ct, ut, st, tt, it = lists(T, fn)
        if umask is not None:
            ut = [u & ~umask[None, :] for u in ut]
        r, f, ev, lat, sat = evaluate(ct, ut, st, tt, it, k)
        if extra:
            r.update(extra)
        out["methods"][name] = {str(b): [r] for b in BUDGETS}       # budget-independent
        print(f"  {name:36s} k {k:4.1f}  recall(<1 ms) {r['recall_evoked_0_1ms']}  "
              f"earliest {r['earliest_recovery_ms']}  FP {r['fp_rate']:.2f}/el/s", flush=True)
        return r, f, ev, lat, sat

    run_flat("no correction", lambda y, p: y)
    run_flat("blanking 1 ms + interpolation", lambda y, p: blank_interp(y, t_samp, 0, 1e-3)[0],
             (t_samp >= 0) & (t_samp < 1e-3))

    # --- physics PER-PULSE (no calibration; provisional wrong model)
    pp_cache = {p: physics_perpulse(data[p][0], data[p][1], Zp, Rp, HP) for p in V + T}
    run_flat("physics per-pulse", lambda y, p: y - pp_cache[p][0],
             extra={"dof": 3, "nominal_params": 3, "params_per_pulse": True})
    # --- SALPA, the original implementation (step 28): per channel, no calibration
    if salpa_available():
        onset = int(round(-T0 / TS))
        peg = (onset, int(np.ceil(2 * PHASE / TS)) + 1)        # the stimulus itself: 0 to 0.45 ms
        for lab, fp in ((SALPA_LABEL, None), (SALPA_PEG_LABEL, peg)):
            sc = {p: salpa_clean(data[p][0], data[p][1], data[p][2], SAT_V, seed=p, forcepeg=fp) for p in V + T}
            cv = [sc[p][0] for p in V]; uv = [sc[p][1] for p in V]
            sv = [data[p][2] for p in V]; tv = [data[p][3] for p in V]
            k = calibrate_k(cv, uv, sv, tv)
            r, *_ = evaluate([sc[p][0] for p in T], [sc[p][1] for p in T], [data[p][2] for p in T],
                             [data[p][3] for p in T], [data[p][1] for p in T], k)
            r.update({"parameters": dict(SALPA_DEFAULTS, forced_peg_samples=fp),
                      "provenance": "PyPI salpa 1.0; github.com/wagenadl/salpa commit 8c06ce7; "
                                    "GPL-3.0-or-later; installed, not copied"})
            out["methods"][lab] = {str(b): [r] for b in BUDGETS}
            print(f"  {lab:48s} k {k:4.1f}  recall(<1 ms) {r['recall_evoked_0_1ms']}  "
                  f"earliest {r['earliest_recovery_ms']}  FP {r['fp_rate']:.2f}/el/s", flush=True)
    elif level == list(LEVELS)[0]:
        print("  SALPA arm SKIPPED: the original package is not installed (pip install salpa; "
              "needs Python >= 3.12)")

    # spike-absorption check (step 22): same test pulses without spikes
    nospk, _ = make_data(HP, level, with_spikes=False, ids=T)
    with_p = np.array([[*pp_cache[p][1], pp_cache[p][2]] for p in T])
    without_fits = {p: physics_perpulse(nospk[p][0], nospk[p][1], Zp, Rp, HP) for p in T}
    without_p = np.array([[*without_fits[p][1], without_fits[p][2]] for p in T])
    shift = np.abs(with_p - without_p).mean(0)
    spread = with_p.std(0, ddof=1)
    names = ["alpha (scale)", "beta (offset)", "C (time constant)"]
    absorb = {nm: {"mean_abs_shift": float(s_), "pulse_to_pulse_sd": float(sd),
                   "shift_over_spread": float(s_ / sd) if sd > 0 else None,
                   "shift_rel_to_value": float(s_ / abs(with_p[:, i].mean()))}
              for i, (nm, s_, sd) in enumerate(zip(names, shift, spread))}
    out["spike_absorption_check"] = absorb
    print("  spike-absorption check (per-pulse physics): " + "; ".join(
        f"{nm.split()[0]} shift {v['shift_rel_to_value']*100:.3f}% of value"
        + (f" = {v['shift_over_spread']:.2f} x pulse-to-pulse SD" if v["shift_over_spread"] is not None else "")
        for nm, v in absorb.items()), flush=True)

    # --- per-pulse physics, stress-test variants (variability none only)
    variants = VARIANTS if level == "none" else [PROVISIONAL_VARIANT]
    if level == "none":
        for var in VARIANTS:
            ret, sf, z1k = var
            if var == PROVISIONAL_VARIANT:
                continue
            Zh = fields[ret]["transfer_ohm"][el] / sf; Rh = fields[ret]["R_solution_ohm"] / sf
            cache = {p: physics_perpulse(data[p][0], data[p][1], Zh, Rh, HP)[0] for p in V + T}
            cv, uv, sv, tv, iv = lists(V, lambda y, p: y - cache[p])
            k = calibrate_k(cv, uv, sv, tv)
            r, *_ = evaluate(*lists(T, lambda y, p: y - cache[p]), k)
            key = f"physics per-pulse | {ret}, sigma x{sf}, {z1k/1e3:.0f} kOhm"
            out["physics_variants"][key] = {str(b): [r] for b in BUDGETS}
        out["physics_variants"][f"physics per-pulse | {PROVISIONAL_VARIANT[0]}, sigma x{PROVISIONAL_VARIANT[1]}, "
                                f"{PROVISIONAL_VARIANT[2]/1e3:.0f} kOhm"] = out["methods"]["physics per-pulse"]

    # --- calibrated arms, budget x draws
    for b in BUDGETS:
        for d_i, cal in enumerate(draws[b]):
            # calibration mean over the pulses in which each sample was usable
            # (a sample clipped in one calibration pulse still uses the others);
            # never-usable samples get 0 and are excluded from fits (U_any)
            Ys = np.array([data[p][0] for p in cal]); Us = np.array([data[p][1] for p in cal])
            cnt = Us.sum(0)
            U_all = cnt > 0
            M = np.where(U_all, (Ys * Us).sum(0) / np.maximum(cnt, 1), 0.0)
            tmpl = M.copy()
            cv, uv, sv, tv, iv = lists(V, lambda y, p: y - tmpl)
            k = calibrate_k(cv, uv, sv, tv)
            r, *_ = evaluate(*lists(T, lambda y, p: y - tmpl), k)
            r["dof"] = n_el * n_t; r["nominal_params"] = n_el * n_t
            out["methods"].setdefault("per-channel average", {}).setdefault(str(b), []).append(r)
            # low-rank, fixed and per-pulse: rank chosen on validation for each
            Uf, sf_, Vtf = np.linalg.svd(M, full_matrices=False)
            for arm in ("low-rank fixed", "low-rank per-pulse"):
                best = None
                for rk in range(1, 21):
                    if arm == "low-rank fixed":
                        Mr = (Uf[:, :rk] * sf_[:rk]) @ Vtf[:rk]
                        fn = lambda y, p, Mr=Mr: y - Mr
                    else:
                        Ur = Uf[:, :rk]
                        fn = lambda y, p, Ur=Ur: y - perpulse_project(y, data[p][1], Ur)
                    cv, uv, sv, tv, iv = lists(V, fn)
                    kk = calibrate_k(cv, uv, sv, tv)
                    rv, *_ = evaluate(cv, uv, sv, tv, iv, kk)
                    sc = rv["recall_evoked_0_1ms"] or 0.0
                    if best is None or sc > best[0]:
                        best = (sc, rk, kk, fn)
                _, rk, kk, fn = best
                r, *_ = evaluate(*lists(T, fn), kk)
                r["rank"] = rk
                r["dof"] = rk * (n_el + n_t - rk) if arm == "low-rank fixed" else rk * n_el + rk * n_t
                r["nominal_params"] = r["dof"]
                out["methods"].setdefault(arm, {}).setdefault(str(b), []).append(r)
            # physics fixed (provisional; all variants at level none)
            for var in variants:
                ret, sf, z1k = var
                Zh = fields[ret]["transfer_ohm"][el] / sf
                Rh = fields[ret]["R_solution_ohm"] / sf
                for arm, off in [("physics fixed, scale + tau only", False), ("physics fixed", True)]:
                    model, coef, C = fit_physics(M, U_all, Zh, Rh, C_GRID, t_fine - T0, -T0, SUB,
                                                 PHASE, window=WIN, offset=off, hp_hz=HP, refine=True)
                    tmpl = model
                    cv, uv, sv, tv, iv = lists(V, lambda y, p: y - tmpl)
                    k = calibrate_k(cv, uv, sv, tv)
                    r, f, ev, lat, sat = evaluate(*lists(T, lambda y, p: y - tmpl), k)
                    r["C_fit_nF"] = C * 1e9; r["coef"] = list(map(float, coef))
                    r["dof"] = (2 if off else 1) + 1; r["nominal_params"] = r["dof"]
                    key = f"{arm} | {ret}, sigma x{sf}, {z1k/1e3:.0f} kOhm"
                    out["physics_variants"].setdefault(key, {}).setdefault(str(b), []).append(r)
                    if var == PROVISIONAL_VARIANT:
                        out["methods"].setdefault(arm, {}).setdefault(str(b), []).append(r)
                    if b == BUDGETS[-1] and d_i == 0 and off:
                        be = out["methods"]["blanking 1 ms + interpolation"][str(b)][0]["earliest_recovery_ms"]
                        out["diagnostics"][key] = _diagnose(tmpl, C, Rh, Zh, T, data, f, ev, sat, r, be)
        print(f"  budget {b:2d}: done ({time.time()-t_start:.0f} s)", flush=True)
    results["by_condition"][cond] = out

# ------------------------------------------------------------------ summary
def med_iqr(vals):
    v = np.array([x for x in vals if x is not None], float)
    if len(v) == 0:
        return None, None, None
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))

ORDER = ["no correction", "blanking 1 ms + interpolation", "per-channel average",
         "low-rank fixed", "low-rank per-pulse", "physics fixed", "physics per-pulse", SALPA_LABEL,
         SALPA_PEG_LABEL, "physics fixed, scale + tau only"]
COLORS = {"no correction": "0.6", "blanking 1 ms + interpolation": "#eb6834",
          "per-channel average": "#1baf7a", "low-rank fixed": "#7b4fd0", "low-rank per-pulse": "#c084fc",
          "physics fixed": "#111111", "physics per-pulse": "#2a78d6",
          "physics fixed, scale + tau only": "#9ca3af", SALPA_LABEL: "#c2410c", SALPA_PEG_LABEL: "#f59e0b"}
STYLE = {"low-rank per-pulse": "--", "physics per-pulse": "--", "physics fixed, scale + tau only": ":",
         SALPA_LABEL: "-.", SALPA_PEG_LABEL: "-."}
summary = {"settings": results["settings"], "split": split, "by_condition": {}}
for cond, out in results["by_condition"].items():
    S = {}
    print(f"\n===== {cond}: recall of evoked spikes < 1 ms, at <= {FP_TARGET:g} false/el/s "
          f"(median [IQR] over {N_DRAWS} draws)")
    print(f"{'arm':48s} " + " ".join(f"{'b=' + str(b):>17s}" for b in BUDGETS) + "  earliest(b=max)")
    for m in ORDER + [k for k in out["methods"] if k not in ORDER]:
        if m not in out["methods"]:
            continue
        S[m], row = {}, []
        for b in BUDGETS:
            rs = out["methods"][m][str(b)]
            md, lo, hi = med_iqr([r["recall_evoked_0_1ms"] for r in rs])
            S[m][str(b)] = {"recall_median": md, "recall_iqr": [lo, hi],
                            "earliest_median_ms": med_iqr([r["earliest_recovery_ms"] for r in rs])[0],
                            "fraction_draws_recall_ge_0.9": float(np.mean([(r["recall_evoked_0_1ms"] or 0) >= 0.9
                                                                          for r in rs])),
                            "dof_median": med_iqr([r.get("dof", 0) for r in rs])[0],
                            "rank_median": med_iqr([r.get("rank") for r in rs])[0]}
            row.append(f"{md:.2f} [{lo:.2f},{hi:.2f}]" if md is not None else "-")
        S[m]["first_budget_median_recall_ge_0.9"] = next(
            (b for b in BUDGETS if (S[m][str(b)]["recall_median"] or 0) >= 0.9), None)
        print(f"{m:48s} " + " ".join(f"{c:>17s}" for c in row)
              + f"  {S[m][str(BUDGETS[-1])]['earliest_median_ms']}")
    adv = {}
    for pair, (ph, em) in {"fixed vs fixed": ("physics fixed", "low-rank fixed"),
                           "per-pulse vs per-pulse": ("physics per-pulse", "low-rank per-pulse")}.items():
        adv[pair] = {str(b): (S[ph][str(b)]["recall_median"] or 0) - (S[em][str(b)]["recall_median"] or 0)
                     for b in BUDGETS}
        print(f"  physics advantage, {pair}: " + ", ".join(f"b={b} {adv[pair][str(b)]:+.2f}" for b in BUDGETS))
    for lab in (SALPA_LABEL, SALPA_PEG_LABEL):
        if lab in S:
            print(f"  physics per-pulse minus {lab}: "
                  f"{(S['physics per-pulse'][str(BUDGETS[0])]['recall_median'] or 0) - (S[lab][str(BUDGETS[0])]['recall_median'] or 0):+.2f}")
    print("  spike absorption: " + "; ".join(f"{k.split()[0]} {v['shift_over_spread']:.2f} x SD"
                                              for k, v in out["spike_absorption_check"].items()
                                              if v["shift_over_spread"] is not None))
    PV = {}
    for key, byb in out["physics_variants"].items():
        a = med_iqr([r["recall_evoked_0_1ms"] for r in byb[str(BUDGETS[0])]])[0]
        z = med_iqr([r["recall_evoked_0_1ms"] for r in byb[str(BUDGETS[-1])]])[0]
        PV[key] = {"recall_b_min": a, "recall_b_max": z, "diagnosis": out["diagnostics"].get(key, {})}
    if len(PV) > 2:
        print("  model-error stress test (recall at b=min -> b=max; diagnosis):")
        for key, v in PV.items():
            print(f"    {key:66s} {v['recall_b_min']:.2f} -> {v['recall_b_max']:.2f}  "
                  f"{v['diagnosis'].get('attribution', '')}")
    summary["by_condition"][cond] = {"methods": S, "physics_advantage": adv, "physics_variants": PV,
                                     "spike_absorption_check": out["spike_absorption_check"]}

# where does the physics advantage disappear? (step 27)
summary["advantage_disappears_at"] = {}
for HP in HP_LIST:
    for pair in ("fixed vs fixed", "per-pulse vs per-pulse"):
        lv = None
        for level in LEVELS:
            a = summary["by_condition"][f"HP {HP or 'off'} | variability {level}"]["physics_advantage"][pair]
            if all(v <= 0 for v in a.values()):
                lv = level; break
        summary["advantage_disappears_at"][f"HP {HP or 'off'} | {pair}"] = lv
        print(f"physics advantage ({pair}, HP {HP or 'off'}) disappears at variability: {lv or 'not within the sweep'}")

# figures: one per filter setting, one panel per variability level
for HP in HP_LIST:
    fig, axs = plt.subplots(1, len(LEVELS), figsize=(4.6 * len(LEVELS), 4.4), sharey=True, layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, level in zip(axs, LEVELS):
        S = summary["by_condition"][f"HP {HP or 'off'} | variability {level}"]["methods"]
        for m in ORDER:
            if m not in S:
                continue
            md = [S[m][str(b)]["recall_median"] for b in BUDGETS]
            ax.plot(BUDGETS, [np.nan if v is None else v for v in md], STYLE.get(m, "-"), marker="o", ms=3.5,
                    color=COLORS[m], lw=2, label=m)
        ax.axhline(0.9, color="0.5", lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        a_j, c_j, d_v = LEVELS[level]
        ax.set_title(f"variability {level}: amp {a_j*100:.0f}%, tau {c_j*100:.0f}%, drift {d_v*1e6:.0f} uV",
                     fontsize=9)
        ax.set_xlabel("calibration pulses"); ax.grid(True, color="0.92", lw=0.5)
    axs[0].set_ylabel(f"evoked spikes found < 1 ms (<= {FP_TARGET:g} false/el/s)")
    axs[0].legend(frameon=False, fontsize=7.5); axs[0].set_ylim(-0.02, 1.02)
    fig.suptitle(f"Aim 2: recall vs calibration budget, high-pass {HP or 'off'} Hz, +-{RANGES_mV[0]} mV; "
                 f"synthetic, generated noise" + (" -- QUICK" if QUICK else ""))
    fig.savefig(FIG / f"budget_variability_HP{HP or 'off'}.png", dpi=180)
    plt.close(fig)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
(OUT / "full_results.json").write_text(json.dumps(results, default=float))
print(f"\ntotal {time.time()-t_start:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
