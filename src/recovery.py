"""Aim 2: recovering spikes under a stimulation artifact.

Recording model (one stimulation trial, many electrodes):
    input_k(t) = artifact_k(t) + spikes_k(t) + noise_k(t)
    amplifier: clips at +-sat; reports the average over each 50 us sample
               (20 kHz, MaxOne). A sample is SATURATED if clipping occurred
               in it (or within `recover` samples before) -- those samples
               are unusable by any method.

Artifact model: artifact_k(t) = Z_k * i(t), with Z_k the transfer resistance
from the solver and i(t) the interface-circuit current (interface.py).

Cleaning methods (all return a cleaned signal and a mask of usable samples):
    blank             discard everything from pulse onset to onset + T_blank
    average_template  subtract the mean over trials at each electrode (needs
                      many trials; anything time-locked to the pulse --
                      including evoked spikes -- is averaged in too)
    physics_template  subtract (alpha * Zhat_k + beta) * ihat(t; tau): the
                      spatial pattern Zhat comes from the solver (possibly
                      from a wrong return path or conductivity). Fitted per
                      trial, using all electrodes at once: alpha (overall
                      scale), beta (a uniform offset -- moving a distant return
                      electrode mostly shifts every electrode's potential by
                      the same amount; found Nov 10, 2026) and tau (interface
                      time constant)

    physics shape +   keep the physics TIME course (interface circuit, tau fitted
    electrode gains   as above) but let the data set each electrode's size:
                      gain_k from the ONSET edge of the pulse (first 100 us,
                      averaged over trials: before any neuron can respond, so
                      evoked spikes cannot leak in); saturated electrodes fall
                      back to the
                      solver's alpha * Zhat_k + beta. Needed when the return path
                      is not known (found Nov 10, 2026: smooth corrections --
                      offset + harmonics -- cannot absorb a periphery-ring
                      return; the mismatch is 0.72-1.19 x at the far edge).

Detection: a negative threshold crossing at -k x noise (noise estimated from
the pre-pulse baseline with the median absolute deviation), one event per
trough; a planted spike counts as found if a detection lies within +-2
samples of its trough.
"""
import numpy as np

from interface import biphasic, circuit_current

FS = 20e3                     # MaxOne sampling rate (Hz)
TS = 1 / FS                   # 50 us


def spike_shape(t_ms):
    """Generic extracellular spike, peak -1 (scale by amplitude): a sharp
    negative trough followed by a slower positive phase, ~1 ms long."""
    return -(np.exp(-0.5 * (t_ms / 0.10) ** 2) - 0.35 * np.exp(-0.5 * ((t_ms - 0.35) / 0.25) ** 2))


def bin_average(sig, sub):
    """Average consecutive blocks of `sub` fine samples (last axis)."""
    n = sig.shape[-1] // sub
    return sig[..., :n * sub].reshape(*sig.shape[:-1], n, sub).mean(-1)


def unit_current_fine(R_sol, C, t_fine_s, t_on_s, phase_s=200e-6, R_buf=0.0):
    return circuit_current(t_fine_s, biphasic(1.0, phase_s, t_on_s), R_sol, C, R_buf)


def highpass(x, fs_fine, hp_hz):
    """First-order high-pass (AC coupling) along the last axis; hp_hz None = off.
    Discrete RC: y[n] = a (y[n-1] + x[n] - x[n-1]), a = tau / (tau + dt)."""
    if not hp_hz:
        return x
    from scipy.signal import lfilter
    tau = 1 / (2 * np.pi * hp_hz)
    a = tau / (tau + 1 / fs_fine)
    return lfilter([a, -a], [1, -a], x, axis=-1)


def record(art_fine, spikes_fine, noise_V, sat_V, sub, recover, rng, hp_hz=None):
    """Amplifier: clip at +-sat (front end), optional first-order high-pass
    (hp_hz), average to 20 kHz, add noise, flag saturation.
    art_fine, spikes_fine: (n_el, n_fine) volts. Returns (samples, saturated)."""
    x = art_fine + spikes_fine
    clipped = np.abs(x) > sat_V
    x = np.clip(x, -sat_V, sat_V)
    x = highpass(x, FS * sub, hp_hz)
    y = bin_average(x, sub)
    sat = bin_average(clipped.astype(float), sub) > 0
    if recover > 0:
        s = sat.copy()
        for k in range(1, recover + 1):
            s[:, k:] |= sat[:, :-k]
        sat = s
    y = y + rng.normal(0, noise_V, y.shape)
    return y, sat


def noise_level(y, baseline):
    """Robust noise estimate (median absolute deviation / 0.6745) per electrode."""
    b = y[..., baseline]
    return np.median(np.abs(b - np.median(b, axis=-1, keepdims=True)), axis=-1) / 0.6745


def detect(clean, usable, sigma, k=5.0):
    """Indices (electrode, sample) of negative threshold crossings, one per trough."""
    thr = -k * sigma[:, None]
    below = (clean < thr) & usable
    out = []
    for e in range(clean.shape[0]):
        idx = np.flatnonzero(below[e])
        if len(idx) == 0:
            continue
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        for g in groups:
            out.append((e, g[np.argmin(clean[e, g])]))
    return out


def harmonic_basis(x, y, order=2):
    """Smooth spatial shapes for the potential a DISTANT source adds across the
    array. Such a potential obeys Laplace's equation and is smooth over the
    array, so low-order harmonic polynomials describe it:
    order 0: 1;  order 1: + x, y;  order 2: + x^2 - y^2, x y.
    Coordinates are scaled to the array size (columns of the returned array)."""
    s = max(np.ptp(x), np.ptp(y)) / 2
    u, v = (x - np.mean(x)) / s, (y - np.mean(y)) / s
    cols = [np.ones_like(u)]
    if order >= 1:
        cols += [u, v]
    if order >= 2:
        cols += [u ** 2 - v ** 2, u * v]
    return np.column_stack(cols)


def fit_physics(y, usable, Zhat, R_sol, C_grid, t_fine_s, t_on_s, sub, phase_s=200e-6,
                window=None, offset=True, extra=None, hp_hz=None, refine=False):
    """Fit artifact = (alpha * Zhat_k + beta + sum_j gamma_j B_jk) * <i(t; C)>_bin
    by least squares over all usable samples of all electrodes in `window`;
    C (with R_sol it sets the time constant) by search over C_grid, and, if
    refine=True, then by a bounded 1-D minimization between the neighbouring
    grid points (needed when tau varies by less than the grid step). Two
    passes: the second ignores samples with large residuals (spikes).
    offset=False drops beta. extra: (n_electrodes, k) further spatial shapes,
    e.g. harmonic_basis(...) (if given, it should not repeat the constant)."""
    if window is None:
        window = np.ones(y.shape[1], dtype=bool)
    m = usable & window[None, :]

    def at(C):
        ib = unit_current_binned(R_sol, C, t_fine_s, t_on_s, sub, phase_s, hp_hz)
        basis = [Zhat[:, None] * ib[None, :]]
        if offset:
            basis.append(np.ones_like(Zhat)[:, None] * ib[None, :])
        if extra is not None:
            basis += [extra[:, j][:, None] * ib[None, :] for j in range(extra.shape[1])]
        keep = m.copy()
        for _ in range(2):
            A = np.column_stack([b[keep] for b in basis])
            coef, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
            model = sum(c * b for c, b in zip(coef, basis))
            res = y - model
            s = np.median(np.abs(res[m])) / 0.6745 + 1e-30
            keep = m & (np.abs(res) < 4 * s)
        return np.sum(res[keep] ** 2) / max(keep.sum(), 1), coef, model

    C_grid = np.asarray(C_grid, float)
    vals = [at(C) for C in C_grid]
    i = int(np.argmin([v[0] for v in vals]))
    sse, coef, model = vals[i]
    C = C_grid[i]
    if refine and len(C_grid) > 2:
        from scipy.optimize import minimize_scalar
        lo, hi = np.log(C_grid[max(i - 1, 0)]), np.log(C_grid[min(i + 1, len(C_grid) - 1)])
        r = minimize_scalar(lambda lc: at(np.exp(lc))[0], bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-4})
        s2, c2, m2 = at(np.exp(r.x))
        if s2 <= sse:
            sse, coef, model, C = s2, c2, m2, float(np.exp(r.x))
    return model, coef, C


def unit_current_binned(R_sol, C, t_fine_s, t_on_s, sub, phase_s=200e-6, hp_hz=None):
    """Physics time course of the artifact (unit drive), as the amplifier
    samples it (through the same high-pass, if any)."""
    i = unit_current_fine(R_sol, C, t_fine_s, t_on_s, phase_s)[None, :]
    return bin_average(highpass(i, FS * sub, hp_hz), sub)[0]


def electrode_gains(y, usable, ib, sample_mask, fallback):
    """Per-electrode artifact size, given the physics time course ib(t):
    least-squares gain sum(y ib) / sum(ib^2) over the usable samples inside
    `sample_mask` -- normally only the ONSET edge of the pulse (before any
    neuron can have responded, so an evoked spike cannot leak into the gain).
    Electrodes with no usable onset sample (saturated) get `fallback` (the
    global physics template's alpha * Zhat_k + beta). Returns (gain, measured)."""
    w = ib[None, :] * (usable & sample_mask[None, :])
    den = np.sum(w * ib[None, :], -1)
    ok = den > 1e-3 * np.max(ib[sample_mask] ** 2) * 1.0
    g = np.where(ok, np.sum(y * w, -1) / np.where(ok, den, 1.0), fallback)
    return g, ok


# ---------------------------------------------------------------------------
# Additions for the calibration-budget study (research plan steps 19-30)
# ---------------------------------------------------------------------------

def _g(t, mu, s):
    return np.exp(-0.5 * ((t - mu) / s) ** 2)


# Spike waveform library (step 19). PARAMETRIC shapes for five waveform
# classes commonly described for extracellular recordings (narrow and broad
# somatic spikes, triphasic axonal, positive-first, small slow). Each has its
# negative trough at t = 0 with depth 1. They are approximations of the
# classes, NOT digitized published templates -- replace or confirm against
# published waveforms before submission, as the plan requires.
WAVEFORMS = {
    "narrow biphasic":  lambda t: -_g(t, 0, 0.07) + 0.40 * _g(t, 0.25, 0.12),
    "broad biphasic":   lambda t: -_g(t, 0, 0.12) + 0.30 * _g(t, 0.45, 0.25),
    "triphasic axonal": lambda t: 0.25 * _g(t, -0.15, 0.06) - _g(t, 0, 0.06) + 0.35 * _g(t, 0.20, 0.10),
    "positive-first":   lambda t: 0.60 * _g(t, -0.18, 0.08) - _g(t, 0, 0.08) + 0.15 * _g(t, 0.3, 0.2),
    "small slow":       lambda t: -_g(t, 0, 0.20) + 0.20 * _g(t, 0.60, 0.35),
}


def detect_fast(clean, usable, sigma, k):
    """Vectorized detect(): (electrode, sample) of the minimum of every run of
    consecutive usable samples below -k * sigma. Same rule as detect()."""
    below = (clean < -k * sigma[:, None]) & usable
    n_el, n = clean.shape
    b = np.zeros((n_el, n + 2), dtype=np.int8)
    b[:, 1:-1] = below
    d = np.diff(b, axis=1)
    st = np.argwhere(d == 1)
    en = np.argwhere(d == -1)
    if len(st) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    lengths = en[:, 1] - st[:, 1]
    L = int(lengths.max())
    cols = st[:, 1][:, None] + np.arange(L)[None, :]
    valid = np.arange(L)[None, :] < lengths[:, None]
    vals = np.where(valid, clean[st[:, 0][:, None], np.minimum(cols, n - 1)], np.inf)
    return st[:, 0], st[:, 1] + np.argmin(vals, axis=1)


def match(events_el, events_s, truth, tol=2):
    """Match detections to planted spikes (truth: list of (el, sample, info)).
    A detection matches a spike on the same electrode within +-tol samples;
    each detection matches at most one spike. Returns (found mask over truth,
    number of unmatched detections)."""
    if len(truth) == 0:
        return np.zeros(0, bool), len(events_el)
    key = events_el.astype(np.int64) * 10_000_000 + events_s
    order = np.argsort(key)
    key_sorted = key[order]
    used = np.zeros(len(key), bool)
    found = np.zeros(len(truth), bool)
    for i, (e, s) in enumerate((tt[0], tt[1]) for tt in truth):
        lo = np.searchsorted(key_sorted, e * 10_000_000 + s - tol)
        hi = np.searchsorted(key_sorted, e * 10_000_000 + s + tol, side="right")
        for j in range(lo, hi):
            if not used[order[j]]:
                used[order[j]] = True
                found[i] = True
                break
    return found, int((~used).sum())


def lowrank(M, r):
    """Best rank-r approximation of M (electrodes x samples) by SVD."""
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    r = min(r, len(s))
    return (U[:, :r] * s[:r]) @ Vt[:r]


def blank_interp(y, t_samp, t0, t1):
    """Blanking with documented interpolation: samples in [t0, t1) replaced by
    a straight line between the last sample before and the first after.
    Returns (signal, mask of samples that were blanked)."""
    m = (t_samp >= t0) & (t_samp < t1)
    out = y.copy()
    idx = np.flatnonzero(m)
    if len(idx):
        a, b = max(idx[0] - 1, 0), min(idx[-1] + 1, len(t_samp) - 1)
        w = (t_samp[idx] - t_samp[a]) / max(t_samp[b] - t_samp[a], 1e-30)
        out[:, idx] = y[:, [a]] * (1 - w) + y[:, [b]] * w
    return out, m


# ---------------------------------------------------------------------------
# SALPA (plan v16 step 28): the ORIGINAL implementation, used directly
# ---------------------------------------------------------------------------
# Provenance: Daniel A. Wagenaar's own code -- PyPI package `salpa` 1.0 (a
# pybind11 wrapper compiled from the author's C++ LocalFit, originally part of
# MEABench), repository https://github.com/wagenadl/salpa (commit 8c06ce7,
# 2026-06-28), licence GPL-3.0-or-later. It is INSTALLED as a dependency
# (`pip install salpa`), not copied into this repository. Reference: Wagenaar
# & Potter (2002), J. Neurosci. Methods 120, 113-120.
# Parameters: the author's command-line defaults from src/salpa.cpp:
# tau (half-width) 3 ms, blank-depeg 20 samples, look-ahead 5 samples,
# asymmetry (chi2) window 10 samples, threshold 3 x the channel's noise SD.
# (The Python wrapper's own defaults differ -- 5, 5, 15 samples; the
# documented command-line tool's defaults are used.) Saturated samples are
# set to the rail, as a saturated amplifier reports them, and the rails are
# the amplifier range, so SALPA sees the same saturation information as every
# other arm. Samples SALPA blanks come back as NaN and are marked unusable.
SALPA_DEFAULTS = {"tau_ms": 3.0, "t_blankdepeg": 20, "t_ahead": 5, "t_chi2": 10, "thresh_std": 3.0}
SALPA_LABEL = "SALPA (original implementation)"
SALPA_PEG_LABEL = "SALPA (original), forced peg over the stimulus"


def salpa_available():
    try:
        import salpa  # noqa: F401
        return True
    except Exception:
        return False


SALPA_PAD = 200            # samples of baseline before and after each snippet (10 ms at 20 kHz)


def salpa_clean(y, usable, sigma, sat_V, fs=FS, seed=0, forcepeg=None):
    """Run the original SALPA on every channel of one pulse. Returns
    (cleaned, usable_after).

    SALPA processes continuous recordings and needs ~4 tau of history before
    it produces output. The synthetic traces are 6 ms snippets, so each is
    embedded in GENERATED baseline noise (the channel's own noise level,
    10 ms before and after), standing in for the surrounding continuous
    recording. The padding carries no information about the artifact.
    forcepeg: (first_sample, n_samples) of the stimulus within the snippet --
    the original tool's forced peg at known stimulus times (-p/-d/-f); None =
    the author's pure defaults (no forced peg)."""
    from salpa import salpa as _salpa
    tau = int(round(SALPA_DEFAULTS["tau_ms"] * 1e-3 * fs))
    rng = np.random.default_rng(seed)
    out = np.empty_like(y)
    for k in range(y.shape[0]):
        x = np.array(y[k], dtype=np.float64)
        x[~usable[k]] = sat_V                       # a saturated amplifier sits on its rail
        pad = lambda: rng.normal(0, float(sigma[k]), SALPA_PAD)
        xx = np.concatenate([pad(), x, pad()])
        kw = {} if forcepeg is None else {"tt_stimuli": [SALPA_PAD + forcepeg[0]], "t_forcepeg": forcepeg[1]}
        yy = _salpa(xx, tau, rail1=-0.999 * sat_V, rail2=0.999 * sat_V,
                    thresh=SALPA_DEFAULTS["thresh_std"] * float(sigma[k]),
                    t_blankdepeg=SALPA_DEFAULTS["t_blankdepeg"], t_ahead=SALPA_DEFAULTS["t_ahead"],
                    t_chi2=SALPA_DEFAULTS["t_chi2"], **kw)
        out[k] = yy[SALPA_PAD:SALPA_PAD + len(x)]
    ok = np.isfinite(out) & usable
    return np.where(np.isfinite(out), out, 0.0), ok
