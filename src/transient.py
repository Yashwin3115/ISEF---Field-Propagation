"""Electrode interface as a constant-phase element (CPE), and the BLIND
prediction of the pulse transient from the impedance sweep (Nov 15, 2026).

Why a CPE and not a capacitor
  A real metal-electrolyte interface is not an ideal capacitor: its impedance
  falls as 1 / (Q (j w)^alpha) with alpha typically 0.7-0.95 (alpha = 1 is the
  ideal capacitor). The sweep is fitted with
      Z_cell(w) = R_s + 1 / (1 / R_ct + Q (j w)^alpha)
  (fit_cell; the ideal-capacitor fit is reported beside it), after the wiring
  measured with the electrode lifted out is removed as a parallel stray path.

The circuit during a pulse
      generator V_g(t) -- R_series --+-- Z_cell (interface + saline) -- return
                                     |
                                  C_stray (wiring + anything clipped to the node)
  current into the saline:  I(w) = V_g(w) Z_p / (R_series + Z_p) / Z_cell,
                            Z_p = Z_cell || 1 / (j w C_stray)
  The saline is resistive, so the probe follows the current with no delay:
      V_probe(t) = (G / sigma) I(t)        G from the committed field prediction.
  What the interface does to the probe trace is therefore (1) a rise time set
  mostly by R_series C_stray and (2) a slow DROOP of the plateau as the CPE
  charges and takes voltage away from the series resistor. Both are predicted
  here, with bands, before any pulse trace is opened.

Method
  A CPE has no ordinary differential equation, so the response is computed in
  the frequency domain: the periodic pulse train's exact Fourier coefficients
  times the transfer function, summed with an inverse FFT. Checked against
  the exact t^alpha law for a current step into a CPE and against an
  independent ODE solution for alpha = 1 (tests/test_transient.py).
"""
import numpy as np
from scipy.optimize import least_squares


def z_cell(f, Rs, Rct, Q, alpha):
    w = 2 * np.pi * np.asarray(f, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = 1.0 / Rct + Q * (1j * w) ** alpha
        z = Rs + np.where(np.abs(y) > 0, 1.0 / y, np.inf)
    return z


def fit_stray(f, Z_lifted):
    """Wiring with the electrode lifted out, as a capacitor (plus a leak):
    Y = 1 / R_leak + j w C. Returns C_stray (median of Im Y / w) and its
    spread across the sweep."""
    f = np.asarray(f, float); Y = 1.0 / np.asarray(Z_lifted, complex)
    C = Y.imag / (2 * np.pi * f)
    return {"C_stray_F": float(np.median(C)),
            "C_stray_rel_sd": float(np.std(C) / abs(np.median(C))) if len(C) > 1 else 0.0,
            "R_leak_ohm": float(1 / max(np.median(Y.real), 1e-15))}


def fit_cell(f, Z, Z_lifted=None, ideal_capacitor=False):
    """Least squares on log|Z| and phase for R_s + (R_ct || CPE). With
    ideal_capacitor=True alpha is fixed at 1 (reported for comparison).
    Returns the parameters, residuals, and the covariance of the fitted
    log-parameters (for the prediction band)."""
    f = np.asarray(f, float); Z = np.asarray(Z, complex)
    if Z_lifted is not None:
        Z = 1.0 / (1.0 / Z - 1.0 / np.asarray(Z_lifted, complex))
    def unpack(p):
        return np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), (1.0 if ideal_capacitor else p[3])
    def res(p):
        m = z_cell(f, *unpack(p))
        return np.concatenate([np.log(np.abs(m)) - np.log(np.abs(Z)), np.angle(m) - np.angle(Z)])
    Rs0 = np.abs(Z[np.argmax(f)])
    w0 = 2 * np.pi * f[np.argmin(f)]
    Q0 = 1.0 / (w0 ** 0.8 * max(np.abs(Z[np.argmin(f)]) - Rs0, 1e-3 * Rs0))
    p0 = [np.log(Rs0), np.log(100 * np.abs(Z[np.argmin(f)])), np.log(Q0)] + ([] if ideal_capacitor else [0.8])
    lo = [-np.inf] * 3 + ([] if ideal_capacitor else [0.3])
    hi = [np.inf] * 3 + ([] if ideal_capacitor else [1.0])
    sol = least_squares(res, p0, bounds=(lo, hi), x_scale="jac")
    Rs, Rct, Q, al = unpack(sol.x)
    n, k = len(sol.fun), len(sol.x)
    s2 = float(np.sum(sol.fun ** 2) / max(n - k, 1))
    JTJ = sol.jac.T @ sol.jac
    cov = np.linalg.pinv(JTJ) * s2
    return {"R_s_ohm": float(Rs), "R_ct_ohm": float(Rct), "Q": float(Q), "alpha": float(al),
            "ideal_capacitor": bool(ideal_capacitor),
            "rms_log_residual": float(np.sqrt(np.mean(sol.fun[:len(f)] ** 2))),
            "rms_phase_residual_deg": float(np.degrees(np.sqrt(np.mean(sol.fun[len(f):] ** 2)))),
            "params_log": [float(v) for v in sol.x], "cov_log": cov.tolist()}


def sample_fits(fit, n, rng):
    """Parameter sets drawn from the fit's covariance (normal in the fitted
    log-parameters; alpha clipped to its bounds)."""
    p = rng.multivariate_normal(fit["params_log"], np.array(fit["cov_log"]), n, check_valid="ignore")
    out = []
    for q in p:
        al = 1.0 if fit["ideal_capacitor"] else float(np.clip(q[3], 0.3, 1.0))
        out.append((float(np.exp(q[0])), float(np.exp(q[1])), float(np.exp(q[2])), al))
    return out


def pulse_train_coefficients(n_harm, period, segments):
    """Exact Fourier coefficients c_k (k = 0..n_harm) of a periodic waveform
    made of rectangular segments [(t_start, t_end, amplitude), ...]."""
    k = np.arange(n_harm + 1)
    w = 2 * np.pi * k / period
    c = np.zeros(n_harm + 1, complex)
    for t1, t2, A in segments:
        with np.errstate(divide="ignore", invalid="ignore"):
            ck = A * (np.exp(-1j * w * t1) - np.exp(-1j * w * t2)) / (1j * w * period)
        ck[0] = A * (t2 - t1) / period
        c += ck
    return c, w


def biphasic_segments(amplitude_V, phase_s, t0=0.0, interphase_s=0.0):
    return [(t0, t0 + phase_s, amplitude_V),
            (t0 + phase_s + interphase_s, t0 + 2 * phase_s + interphase_s, -amplitude_V)]


def pulse_response(segments, period, dt, R_series, cell, C_stray=0.0, t_rise_gen=0.0):
    """Steady-state periodic response of the circuit to a rectangular pulse
    train. cell = (R_s, R_ct, Q, alpha). Returns t (one period), the current
    into the saline I(t) and the electrode-node voltage V_node(t).
    t_rise_gen: the generator's own 10-90% rise time (first-order)."""
    n = int(round(period / dt))
    n += n % 2
    c, w = pulse_train_coefficients(n // 2, period, segments)
    f = w / (2 * np.pi)
    Zc = z_cell(np.where(f > 0, f, 1.0), *cell)
    Zc[0] = cell[0] + cell[1]                                  # DC: the CPE is open
    Yp = 1.0 / Zc + 1j * w * C_stray
    Zp = 1.0 / Yp
    H = Zp / (R_series + Zp)
    if t_rise_gen > 0:
        H = H / (1 + 1j * w * (t_rise_gen / 2.197))
    Vn = c * H
    In = Vn / Zc
    t = np.arange(n) * (period / n)
    return t, np.fft.irfft(In, n) * n, np.fft.irfft(Vn, n) * n


def transient_metrics(t, v, phase_s, t_edge=None):
    """The numbers compared between prediction and measurement, computed the
    SAME way for both:
      level      mean of the trace 15-25% into the first phase (baseline removed)
      rise_s     10% -> 90% of `level` on the first edge (linear interpolation)
      droop      (level - mean at 85-95% of the phase) / level
      t_edge     time the trace first crosses 50% of `level`
    The baseline is the mean of the samples before the pulse. For a measured
    trace t_edge is found from the data; phase_s is the set pulse width."""
    t = np.asarray(t, float); v = np.asarray(v, float)
    if t_edge is None:
        n0 = max(len(v) // 20, 5)
        base = np.mean(v[:n0])
        dv = v - base
        big = np.flatnonzero(np.abs(dv) > 0.5 * np.max(np.abs(dv)))
        k0 = big[0]
        t_edge = t[k0]
    else:
        base = np.mean(v[t < t_edge - 0.02 * phase_s]) if np.any(t < t_edge - 0.02 * phase_s) else 0.0
        dv = v - base
    win = lambda a, b: (t >= t_edge + a * phase_s) & (t <= t_edge + b * phase_s)
    level = float(np.mean(dv[win(0.15, 0.25)]))
    late = float(np.mean(dv[win(0.85, 0.95)]))
    seg = (t >= t_edge - 0.1 * phase_s) & (t <= t_edge + 0.15 * phase_s)
    ts, vs = t[seg], dv[seg] / level
    def cross(th):
        k = np.flatnonzero(vs >= th)
        if len(k) == 0 or k[0] == 0:
            return ts[0] if len(k) else np.nan
        i = k[0]
        return ts[i - 1] + (th - vs[i - 1]) * (ts[i] - ts[i - 1]) / (vs[i] - vs[i - 1])
    t10, t50, t90 = cross(0.1), cross(0.5), cross(0.9)
    return {"level": level, "rise_s": float(t90 - t10), "droop": float((level - late) / level),
            "t_edge_s": float(t50), "baseline": float(base)}
