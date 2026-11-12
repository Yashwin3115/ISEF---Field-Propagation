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
