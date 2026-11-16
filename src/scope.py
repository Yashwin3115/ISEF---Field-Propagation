"""Reading oscilloscope exports and the electrode impedance sweep.

plateau(t, v)       the potential during the FIRST phase of a rectangular
                    pulse: baseline (before the pulse) subtracted, averaged
                    over the middle half of the phase, so the edges (and any
                    ringing) are excluded. Quasi-static conduction makes this
                    flat part equal to I * G / sigma.
read_trace(path)    two numeric columns (time, voltage) from a scope CSV;
                    metadata/header lines that are not two numbers are skipped.
fit_randles(...)    impedance sweep (plan step 56) -> R_s + (R_ct || CPE),
                    with the wiring baseline (electrode lifted out) removed as
                    a parallel stray path.

Scope data loader (Nov 15, 2026) -- so the noise floor in every band is REAL
read_scope_csv(path)   any of the common export layouts (plain time,CH1[,CH2];
                       Rigol "X,CH1,Start,Increment"; Tektronix with metadata in
                       the first columns; Keysight/Siglent header blocks), one or
                       two channels, with whatever settings the file states
quantization(v)        the ADC step found from the data itself, the number of
                       levels used, and the clipped fraction
average_traces(...)    sample-by-sample mean and SD of repeated traces
residual_quantization_error(lsb, noise_sd)
                       the part of the quantization error that averaging does
                       NOT remove: with noise below ~0.5 step, every trace
                       rounds the same way
plateau_reading(...)   plateau, its standard error across traces, and a noise
                       floor that includes the quantization term
"""
import csv
import numpy as np
from scipy.optimize import least_squares


def read_trace(path):
    t, v = [], []
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            nums = []
            for c in row:
                try:
                    nums.append(float(c))
                except ValueError:
                    pass
            if len(nums) >= 2 and len(nums) == len([c for c in row if c.strip()]):
                t.append(nums[0]); v.append(nums[1])
    if len(t) < 10:
        raise ValueError(f"{path}: fewer than 10 numeric (time, voltage) rows found")
    return np.array(t), np.array(v)


def plateau(t, v, frac=0.5):
    """Returns (plateau_V, sd_V, info). The pulse is located as the first
    stretch where |v - baseline| exceeds half its maximum."""
    n0 = max(len(v) // 20, 5)
    base = np.median(v[:n0])
    dv = v - base
    big = np.abs(dv) > 0.5 * np.max(np.abs(dv))
    idx = np.flatnonzero(big)
    if len(idx) == 0:
        raise ValueError("no pulse found")
    start = idx[0]
    sign = np.sign(dv[start])
    amp = np.max(np.abs(dv))
    k = start                                  # end of the first phase:
    while k + 1 < len(v) and sign * dv[k + 1] > 0.25 * amp:   # same sign, still large
        k += 1
    n = k - start + 1
    a, b = start + int(n * (1 - frac) / 2), start + int(n * (1 + frac) / 2)
    seg = dv[a:max(b, a + 1)]
    return float(np.mean(seg)), float(np.std(seg)), {
        "baseline_V": float(base), "phase_start_s": float(t[start]), "phase_end_s": float(t[k]),
        "phase_width_s": float(t[k] - t[start]), "samples_used": len(seg)}


def randles(f, Rs, Rct, Q, alpha):
    """Z = Rs + 1 / (1/Rct + Q (j w)^alpha)   (CPE double layer)."""
    w = 2 * np.pi * np.asarray(f)
    return Rs + 1.0 / (1.0 / Rct + Q * (1j * w) ** alpha)


def impedance_from_sweep(f, V_res, V_el, phase_deg, R_series):
    """|Z| and phase of the electrode from the voltages across the series
    resistor (-> current) and across the electrode; phase of V_el relative
    to the current."""
    I = np.asarray(V_res) / R_series
    return np.asarray(V_el) / I * np.exp(1j * np.radians(phase_deg))


def fit_randles(f, Z, Z_wiring=None):
    """Least squares on log|Z| and phase. Z_wiring: impedance measured with
    the electrode lifted out (a parallel stray path), removed first."""
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    if Z_wiring is not None:
        Z = 1.0 / (1.0 / Z - 1.0 / np.asarray(Z_wiring, complex))
    def res(p):
        Rs, Rct, Q, al = np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3]
        m = randles(f, Rs, Rct, Q, al)
        return np.concatenate([np.log(np.abs(m)) - np.log(np.abs(Z)),
                               np.angle(m) - np.angle(Z)])
    Rs0 = np.abs(Z[np.argmax(f)])
    p0 = [np.log(Rs0), np.log(10 * np.abs(Z[np.argmin(f)])), np.log(1e-7), 0.8]
    sol = least_squares(res, p0, bounds=([-np.inf, -np.inf, -np.inf, 0.3], [np.inf, np.inf, np.inf, 1.0]))
    Rs, Rct, Q, al = np.exp(sol.x[0]), np.exp(sol.x[1]), np.exp(sol.x[2]), sol.x[3]
    return {"R_s_ohm": float(Rs), "R_ct_ohm": float(Rct), "Q": float(Q), "alpha": float(al),
            "rms_log_residual": float(np.sqrt(np.mean(sol.fun[:len(f)] ** 2))),
            "rms_phase_residual_deg": float(np.degrees(np.sqrt(np.mean(sol.fun[len(f):] ** 2))))}


# ---------------------------------------------------------------------------
# scope data loader (Nov 15, 2026)
# ---------------------------------------------------------------------------

def _num(c):
    try:
        return float(c)
    except (TypeError, ValueError):
        return None


def _split(line):
    for sep in (",", ";", "\t"):
        if sep in line:
            return [c.strip() for c in line.rstrip("\r\n").split(sep)]
    return line.split()


def read_scope_csv(path):
    """Read an oscilloscope CSV export. Returns
        {"t": seconds, "channels": {name: volts}, "meta": {...}, "format": str}
    Layouts handled (the school's scope model is not known yet):
      plain     time, CH1[, CH2] with or without header lines
      rigol     header "X,CH1[,CH2],Start,Increment"; the first data row carries
                the start time and the sample interval, X is the sample index
      tek       Tektronix TBS/TDS: settings in columns 1-2 of the first rows,
                time and volts in columns 4-5 of every row
      block     Keysight / Siglent: lines of settings, then "Second,Volt" data
    The raw file is never modified."""
    with open(path, newline="", errors="replace") as fh:
        lines = [ln for ln in fh.read().splitlines() if ln.strip()]
    rows = [_split(ln) for ln in lines]
    meta = {}
    low = [[c.lower() for c in r] for r in rows[:40]]
    # --- Rigol: X,CH1,(CH2,)Start,Increment
    for k, r in enumerate(low):
        if "start" in r and "increment" in r and r and r[0] in ("x", "time"):
            i_s, i_i = r.index("start"), r.index("increment")
            names = [rows[k][j] for j in range(1, i_s) if rows[k][j]]
            data = [q for q in rows[k + 1:] if _num(q[0]) is not None]
            start, inc = _num(data[0][i_s]), _num(data[0][i_i])
            idx = np.array([_num(q[0]) for q in data])
            ch = {n: np.array([_num(q[1 + j]) if len(q) > 1 + j and _num(q[1 + j]) is not None else np.nan
                               for q in data]) for j, n in enumerate(names)}
            meta.update(start_s=start, increment_s=inc)
            return {"t": start + (idx - idx[0]) * inc, "channels": ch, "meta": meta, "format": "rigol"}
    # --- Tektronix: text in column 0 of the first rows, numbers in columns 3 and 4
    tek = [r for r in rows if len(r) >= 5 and _num(r[3]) is not None and _num(r[4]) is not None]
    if len(tek) >= 0.9 * len(rows) and any(r[0] and _num(r[0]) is None for r in rows[:20]):
        for r in rows[:25]:
            if r[0] and _num(r[0]) is None:
                meta[r[0]] = r[1] if len(r) > 1 else ""
        name = str(meta.get("Source", "CH1")) or "CH1"
        return {"t": np.array([_num(r[3]) for r in tek]),
                "channels": {name: np.array([_num(r[4]) for r in tek])}, "meta": meta, "format": "tek"}
    # --- plain / block: the longest run of rows that are all numbers
    numeric = [all(_num(c) is not None for c in r if c != "") and sum(c != "" for c in r) >= 2 for r in rows]
    first = next((k for k in range(len(rows)) if numeric[k] and all(numeric[k:k + 5])), None)
    if first is None:
        raise ValueError(f"{path}: no block of numeric (time, voltage) rows found")
    for r in rows[:first]:
        if len(r) >= 2 and r[0] and _num(r[0]) is None and _num(r[1]) is not None:
            meta[r[0].rstrip(":")] = _num(r[1])
    data = [[_num(c) for c in r if c != ""] for r, ok in zip(rows[first:], numeric[first:]) if ok]
    ncol = min(len(d) for d in data)
    arr = np.array([d[:ncol] for d in data], float)
    names = None
    if first > 0:
        h = [c for c in rows[first - 1] if c != ""]
        if len(h) >= ncol and all(_num(c) is None for c in h[:ncol]):
            names = h[1:ncol]
            if names and all(n.lower() in ("volt", "v", "value", "volts") for n in names) and first > 1:
                h2 = [c for c in rows[first - 2] if c != ""]
                if len(h2) >= ncol:
                    names = [f"CH{c}" if _num(c) is not None else c for c in h2[1:ncol]]
    if not names or len(set(names)) != ncol - 1:
        names = [f"CH{j}" for j in range(1, ncol)]
    if len(arr) < 10:
        raise ValueError(f"{path}: fewer than 10 numeric rows found")
    return {"t": arr[:, 0], "channels": {n: arr[:, 1 + j] for j, n in enumerate(names)},
            "meta": meta, "format": "plain" if first <= 2 else "block"}


def quantization(v):
    """The ADC step (LSB) found from the data: the smallest spacing between
    the distinct values present. Returns lsb_V (None if the values are not on a
    grid -- the scope averaged or used high-resolution mode before exporting),
    levels_used, bits_used (log2 of range / lsb) and the fraction of samples
    sitting on the top or bottom level (clipping if more than a few).
    lsb_reliable is False when fewer than 6 levels are present: a very quiet
    trace may never touch two NEIGHBOURING levels, and then the smallest
    spacing seen is a multiple of the true step. Give the V/div setting then."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    u = np.unique(v)
    out = {"n_samples": int(len(v)), "levels_used": int(len(u)), "lsb_V": None, "bits_used": None,
           "lsb_reliable": False,
           "clipped_fraction": float(np.mean((v == u[0]) | (v == u[-1]))) if len(u) > 1 else 1.0}
    if len(u) < 2:
        return out
    d = np.diff(u)
    lsb = float(np.min(d))
    steps = d / lsb
    on_grid = np.all(np.abs(steps - np.round(steps)) < 0.05) and len(u) <= 70000 and len(u) < 0.5 * len(v)
    if on_grid:
        out["lsb_V"] = lsb
        out["bits_used"] = float(np.log2((u[-1] - u[0]) / lsb + 1))
        out["lsb_reliable"] = len(u) >= 6
    return out


def lsb_from_settings(volts_per_div, bits=8, divisions=8, full_scale_divisions=None):
    """ADC step expected from the vertical setting: most 8-bit scopes spread
    their 256 levels over slightly more than the visible divisions (often 10
    for 8 visible); pass full_scale_divisions if the manual gives it."""
    fs = full_scale_divisions if full_scale_divisions else divisions
    return volts_per_div * fs / 2 ** bits


def residual_quantization_error(lsb, noise_sd):
    """RMS error of the MEAN of many quantized samples that averaging cannot
    remove. A quantizer with Gaussian noise of SD s ahead of it has a mean
    error that depends on where the true value sits between two levels:
    a sawtooth whose k-th harmonic (amplitude lsb / (pi k)) is damped by
    exp(-2 pi^2 k^2 s^2 / lsb^2). Its rms over all positions is
        sqrt( sum_k (lsb / (pi k))^2 / 2 * exp(-4 pi^2 k^2 (s / lsb)^2) ),
    which is lsb / sqrt(12) with no noise (plain rounding) and below 0.2% of
    a step once s >= 0.5 lsb: averaging then works fully."""
    if not lsb:
        return 0.0
    k = np.arange(1, 200)
    r2 = np.sum((lsb / (np.pi * k)) ** 2 / 2 * np.exp(-4 * np.pi ** 2 * k ** 2 * (noise_sd / lsb) ** 2))
    return float(np.sqrt(r2))


def average_traces(traces):
    """traces: list of (t, v) on the same trigger-relative time base. Returns
    t, mean, sample-by-sample SD (ddof 1), n. Traces of different length are
    cut to the shortest; a time base that differs by more than half a sample
    raises (those traces must not be averaged together)."""
    n = min(len(t) for t, _ in traces)
    t0 = np.asarray(traces[0][0][:n], float)
    dt = np.median(np.diff(t0))
    for t, _ in traces[1:]:
        if np.max(np.abs(np.asarray(t[:n]) - t0)) > 0.5 * dt:
            raise ValueError("traces are on different time bases (different trigger position or "
                             "time/div): do not average them together")
    V = np.array([np.asarray(v[:n], float) for _, v in traces])
    sd = V.std(axis=0, ddof=1) if len(V) > 1 else np.zeros(n)
    return t0, V.mean(axis=0), sd, len(V)


def effective_samples(x):
    """Number of statistically independent samples in a noise record: n divided
    by the integrated autocorrelation time (1 for white noise). Scope noise is
    band-limited and often carries hum, so neighbouring samples are not
    independent and a window mean improves by less than sqrt(n)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if n < 16 or np.std(x) == 0:
        return float(n)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n // 2]
    ac = ac / ac[0]
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] < 0.05:
            break
        tau += 2 * ac[k]
    return float(max(n / tau, 1.0))


def plateau_reading(traces, lsb=None, frac=0.5):
    """One reading for the measurements CSV from repeated traces of the same
    pulse at the same position.
      plateau_V      mean of the per-trace plateaus (baseline subtracted)
      plateau_V_sd   SD across traces (ddof 1); 0 if only one trace
      standard_error random part: sd / sqrt(n); for ONE trace (e.g. the scope
                     averaged internally) it is the within-plateau sample SD
                     divided by sqrt(effective independent samples), combined
                     with the same for the baseline
      quantization_V the part averaging cannot remove (both levels: x sqrt 2)
      noise_floor_V  sqrt(standard_error^2 + quantization_V^2) -- the number
                     for the noise_floor_V column
    lsb: ADC step in volts; found from the data when the samples are on a grid."""
    vals, sds, infos, base_sd, neff = [], [], [], [], []
    for t, v in traces:
        pv, sd, info = plateau(t, v, frac)
        vals.append(pv); sds.append(sd); infos.append(info)
        n0 = max(len(v) // 20, 5)
        b = np.asarray(v[:n0], float)
        base_sd.append(np.std(b)); neff.append(effective_samples(b) / len(b))
    vals = np.array(vals)
    q = quantization(np.concatenate([np.asarray(v, float) for _, v in traces]))
    lsb_from = "given"
    if lsb is None:
        lsb = q["lsb_V"]
        lsb_from = "data" if q["lsb_reliable"] else ("data, too few levels: give --vdiv" if lsb else "none")
    noise_sd = float(np.median(base_sd))
    n = len(vals)
    if n > 1:
        sd = float(vals.std(ddof=1)); se = sd / np.sqrt(n)
    else:
        sd = 0.0
        m = infos[0]["samples_used"]; n0 = max(len(traces[0][1]) // 20, 5)
        fr = float(np.median(neff))
        se = float(np.hypot(sds[0] / np.sqrt(max(m * fr, 1)), base_sd[0] / np.sqrt(max(n0 * fr, 1))))
    qv = np.sqrt(2) * residual_quantization_error(lsb, noise_sd) if lsb else 0.0
    return {"plateau_V": float(vals.mean()), "plateau_V_sd": sd, "n_traces": n,
            "standard_error_V": float(se), "quantization_V": float(qv),
            "noise_floor_V": float(np.hypot(se, qv)), "lsb_V": lsb, "lsb_from": lsb_from,
            "single_sample_noise_V": noise_sd, "noise_over_lsb": (noise_sd / lsb if lsb else None),
            "levels_used": q["levels_used"], "clipped_fraction": q["clipped_fraction"],
            "phase_width_s": float(np.median([i["phase_width_s"] for i in infos])),
            "independent_fraction": float(np.median(neff)),
            "values_on_adc_grid": q["lsb_V"] is not None}


def traces_needed(single_sample_noise_V, target_V, plateau_samples, independent_fraction=1.0,
                  lsb=None):
    """How many traces must be averaged for the plateau reading to reach
    target_V (1 SD), assuming the noise is random from trace to trace.
    Returns (n_traces, reachable): reachable is False when the quantization
    term alone is already above the target (more averaging cannot help; use a
    smaller V/div, add noise, or raise the current)."""
    m = max(plateau_samples * independent_fraction, 1.0)
    per_trace = single_sample_noise_V * np.sqrt(2.0 / m)       # plateau and baseline windows
    qv = np.sqrt(2) * residual_quantization_error(lsb, single_sample_noise_V) if lsb else 0.0
    if qv >= target_V:
        return None, False
    return int(np.ceil(per_trace ** 2 / (target_V ** 2 - qv ** 2))), True
