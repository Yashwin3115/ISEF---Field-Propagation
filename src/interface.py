"""Electrode-electrolyte interface: from an applied voltage pulse to the
current that actually flows, and from that current to the artifact at every
electrode.

Circuit (in series, from the stimulator to the return):

    V(t) --[ R_buf ]--[ interface: C_dl in parallel with R_ct ]--[ R_sol ]-- 0 V

  V(t)   biphasic voltage pulse, positive phase first (MaxLab Live manual):
         +A for one phase, -A for the next, then 0.
  R_buf  stimulator output resistance. NOT documented -- default 0 (ideal
         source); treat as a sensitivity parameter.
  C_dl   interface capacitance. From the Stimulation Guide's impedance at
         1 kHz, assuming the interface is capacitive there:
             C_dl = 1 / (2 pi f |Z|)
         Check: this reproduces the guide's "injected charge at 200 mV"
         column exactly (e.g. 600 kOhm -> 0.265 nF -> 0.053 nC).
  R_ct   charge-transfer (leak) resistance. Not documented -- default
         infinite (purely capacitive), the usual assumption for Pt black and
         PEDOT at these amplitudes.
  R_sol  resistance of the fluid from the pad to the return: computed by the
         solver (transfer_run's R_solution_ohm).

Because the fluid is linear and responds instantly (quasi-static), the
potential at electrode k is

    phi_k(t) = Z_k * i(t),   Z_k = transfer resistance from the solver (ohm)

so the SHAPE in space comes from the solver and the SHAPE in time from this
circuit.

The circuit is solved exactly: within each constant-voltage stretch the
capacitor voltage relaxes exponentially toward its end value.
"""
import numpy as np


def capacitance_from_impedance(z_ohm, f_hz=1e3):
    return 1.0 / (2 * np.pi * f_hz * z_ohm)


def biphasic(amplitude_V, phase_s, t_start=0.0):
    """List of (time, voltage) steps for a positive-first biphasic pulse."""
    return [(t_start, amplitude_V), (t_start + phase_s, -amplitude_V),
            (t_start + 2 * phase_s, 0.0)]


def circuit_current(t, steps, R_sol, C_dl, R_buf=0.0, R_ct=np.inf):
    """Current i(t) (A) at the times t for a piecewise-constant source.

    steps: list of (time, voltage), sorted; V = 0 before the first step.
    """
    t = np.asarray(t, dtype=float)
    R = R_buf + R_sol
    if np.isinf(R_ct):
        tau, gain = R * C_dl, 1.0
    else:
        tau = C_dl * R * R_ct / (R + R_ct)
        gain = R_ct / (R + R_ct)                 # final capacitor voltage per volt

    # capacitor voltage at the start of each segment
    times = [s[0] for s in steps]
    volts = [s[1] for s in steps]
    vc_start = []
    vc = 0.0
    for k, (tk, Vk) in enumerate(steps):
        vc_start.append(vc)
        t_next = times[k + 1] if k + 1 < len(steps) else np.inf
        if np.isfinite(t_next):
            v_inf = gain * Vk
            vc = v_inf + (vc - v_inf) * np.exp(-(t_next - tk) / tau)

    i = np.zeros_like(t)
    for k, (tk, Vk) in enumerate(steps):
        t_next = times[k + 1] if k + 1 < len(steps) else np.inf
        sel = (t >= tk) & (t < t_next)
        v_inf = gain * Vk
        vc_t = v_inf + (vc_start[k] - v_inf) * np.exp(-(t[sel] - tk) / tau)
        i[sel] = (Vk - vc_t) / R
    return i


def segment_charges(steps, R_sol, C_dl, R_buf=0.0, R_ct=np.inf):
    """Exact charge (C) delivered during each constant-voltage stretch.
    The last stretch runs until the capacitor has fully relaxed.

    Charge into the interface = C * (change in capacitor voltage)
                              + (1 / R_ct) * integral of capacitor voltage.
    """
    R = R_buf + R_sol
    if np.isinf(R_ct):
        tau, gain = R * C_dl, 1.0
    else:
        tau, gain = C_dl * R * R_ct / (R + R_ct), R_ct / (R + R_ct)
    q, vc = [], 0.0
    for k, (tk, Vk) in enumerate(steps):
        L = steps[k + 1][0] - tk if k + 1 < len(steps) else np.inf
        v_inf = gain * Vk
        decay = 0.0 if np.isinf(L) else np.exp(-L / tau)
        ve = v_inf + (vc - v_inf) * decay
        integral_vc = (0.0 if v_inf == 0 else v_inf * L) + (vc - v_inf) * tau * (1 - decay)
        leak = 0.0 if np.isinf(R_ct) else integral_vc / R_ct
        q.append(C_dl * (ve - vc) + leak)
        vc = ve
    return q


def pulse_summary(amplitude_V, phase_s, R_sol, C_dl, R_buf=0.0, R_ct=np.inf, dt=0.05e-6):
    """Peak current, time constant and charge in each phase (charges exact)."""
    steps = biphasic(amplitude_V, phase_s)
    R = R_buf + R_sol
    tau = R * C_dl if np.isinf(R_ct) else C_dl * R * R_ct / (R + R_ct)
    t = np.arange(0, 2 * phase_s + 20 * tau, dt)
    i = circuit_current(t, steps, R_sol, C_dl, R_buf, R_ct)
    q1, q2, q3 = segment_charges(steps, R_sol, C_dl, R_buf, R_ct)
    return {
        "tau_s": tau,
        "peak_current_A": float(np.max(np.abs(i))),
        "charge_phase1_C": q1,
        "charge_phase2_C": q2,
        "charge_after_C": q3,
        "net_charge_C": q1 + q2 + q3,
        "t": t, "i": i,
    }


def sample_average(t, signal, fs=20e3):
    """Average a signal over each sample period (20 kHz on MaxOne -> 50 us).
    This is ONE possible model of what the amplifier reports; the real
    amplifier filter is not documented."""
    T = 1.0 / fs
    k = np.floor(t / T).astype(int)
    n = k.max() + 1
    sums = np.bincount(k, weights=signal, minlength=n)
    counts = np.bincount(k, minlength=n)
    return (np.arange(n) + 0.5) * T, sums / np.maximum(counts, 1)
