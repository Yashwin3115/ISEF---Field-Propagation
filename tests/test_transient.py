"""Checks for the CPE interface model and the blind pulse-transient
prediction (src/transient.py), on made-up circuits with known answers.

1. constant current into a CPE: the voltage across it must follow the exact
   law  V(t) = I t^alpha / (Q Gamma(1 + alpha))  (alpha = 0.8)
2. alpha = 1 with a stray capacitor: the frequency-domain result must match
   an independent time-domain ODE solution of the same circuit
3. fit_cell recovers known CPE parameters from a noisy sweep with a wiring
   path in parallel, and beats the ideal-capacitor fit; fit_stray recovers
   the stray capacitance
4. transient_metrics: rise time of an exact exponential edge = tau ln 9, and a
   known droop is recovered
5. end to end: sweep -> fit -> predicted probe transient lies within 2% of
   the transient computed from the TRUE parameters
"""
import sys
from pathlib import Path
import numpy as np
from math import gamma
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from transient import (z_cell, fit_cell, fit_stray, pulse_response, biphasic_segments,
                       transient_metrics, sample_fits)

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

# 1 current step into a pure CPE (R_series huge -> constant current; R_s = 0)
Q, al, R = 2e-7, 0.8, 1e9
A, ph, T = 1e3, 200e-6, 0.2                       # 1 uA; long period -> no memory of the last pulse
t, I, Vn = pulse_response(biphasic_segments(A, ph, 1e-3), T, 0.5e-6, R, (1e-6, 1e15, Q, al))
m = (t > 1e-3 + 5e-6) & (t < 1e-3 + 0.9 * ph)
exact = (A / R) * (t[m] - 1e-3) ** al / (Q * gamma(1 + al))
got = Vn[m] - np.interp(1e-3 - 2e-6, t, Vn)
err1 = np.max(np.abs(got / exact - 1))
check("CPE charging law t^alpha", err1 < 0.01, f"max error {err1*100:.2f}% over the first phase (alpha {al})")

# 2 alpha = 1: R - [C_stray || (Rs + C)] vs ODE
R, Rs, C, Cs, A, ph, T = 100e3, 650.0, 40e-9, 10e-9, 5.0, 200e-6, 20e-3   # large stray: a visible edge
t, I, Vn = pulse_response(biphasic_segments(A, ph, 1e-3), T, 0.05e-6, R, (Rs, 1e15, C, 1.0), Cs)
def vg(tt):
    tt = tt % T
    return A * ((tt >= 1e-3) & (tt < 1e-3 + ph)) - A * ((tt >= 1e-3 + ph) & (tt < 1e-3 + 2 * ph))
def rhs(tt, y):                                   # y = [V_node, V_C]
    i_cell = (y[0] - y[1]) / Rs
    return [((vg(tt) - y[0]) / R - i_cell) / Cs, i_cell / C]
# start from the periodic state the frequency-domain solution gives at t = 0
vc0 = Vn[0] - I[0] * Rs
te = np.linspace(0.95e-3, 1.5e-3, 1101)
sol = solve_ivp(rhs, [0, 1.5e-3], [Vn[0], vc0], method="LSODA", t_eval=te, rtol=1e-9, atol=1e-12,
                max_step=2e-6)
I_ode = (sol.y[0] - sol.y[1]) / Rs
I_fd = np.interp(te, t, I)
err2 = np.max(np.abs(I_ode - I_fd)) / np.max(np.abs(I_ode))
check("alpha = 1 vs independent ODE", err2 < 0.01, f"max difference {err2*100:.3f}% of the peak current")

# 3 fit
rng = np.random.default_rng(3)
f = np.array([10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
true = (645.0, 5e6, 1.5e-7, 0.82)
Zw = 1 / (1j * 2 * np.pi * f * 60e-12 + 1 / 5e8)
Zm = 1 / (1 / z_cell(f, *true) + 1 / Zw)
Zm = Zm * (1 + rng.normal(0, 0.01, len(f))) * np.exp(1j * np.radians(rng.normal(0, 0.5, len(f))))
fit = fit_cell(f, Zm, Zw)
cap = fit_cell(f, Zm, Zw, ideal_capacitor=True)
st = fit_stray(f, Zw)
check("CPE fit vs ideal capacitor", abs(fit["R_s_ohm"] / true[0] - 1) < 0.03 and abs(fit["alpha"] - true[3]) < 0.01
      and abs(fit["Q"] / true[2] - 1) < 0.1 and cap["rms_log_residual"] > 3 * fit["rms_log_residual"]
      and abs(st["C_stray_F"] / 60e-12 - 1) < 1e-3,
      f"R_s {fit['R_s_ohm']:.0f} (645), alpha {fit['alpha']:.3f} (0.82), Q {fit['Q']:.3g} (1.5e-7); residual "
      f"{fit['rms_log_residual']:.4f} vs capacitor {cap['rms_log_residual']:.4f}; C_stray {st['C_stray_F']*1e12:.1f} pF")

# 4 metrics: an exact exponential edge (no droop), then a fast edge with a known droop
tau, dr, ph = 8e-6, 0.06, 200e-6
tt = np.arange(0, 1e-3, 0.2e-6)
on = (tt >= 0.3e-3) & (tt < 0.3e-3 + ph)
v1 = np.where(on, 1 - np.exp(-(tt - 0.3e-3) / tau), 0)
v2 = np.where(on, (1 - np.exp(-(tt - 0.3e-3) / 0.5e-6)) * (1 - dr * (tt - 0.3e-3) / ph), 0)
m1, m2 = transient_metrics(tt, v1, ph), transient_metrics(tt, v2, ph)
droop_exact = 0.7 * dr / (1 - 0.2 * dr)
check("transient metrics", abs(m1["rise_s"] / (tau * np.log(9)) - 1) < 0.02 and abs(m2["droop"] - droop_exact) < 5e-4,
      f"rise {m1['rise_s']*1e6:.2f} us (tau ln 9 = {tau*np.log(9)*1e6:.2f}); droop {m2['droop']*100:.2f}% "
      f"(exact {droop_exact*100:.2f}%)")

# 5 end to end
R, A, ph, T, Cs = 100e3, 5.0, 200e-6, 10e-3, 60e-12
def probe(cell, cs):
    t_, I_, _ = pulse_response(biphasic_segments(A, ph, 1e-3), T, 0.1e-6, R, cell, cs)
    return t_, I_
t_, I_true = probe(true, 60e-12)
_, I_fit = probe((fit["R_s_ohm"], fit["R_ct_ohm"], fit["Q"], fit["alpha"]), st["C_stray_F"])
m_true, m_fit = transient_metrics(t_, I_true, ph, 1e-3), transient_metrics(t_, I_fit, ph, 1e-3)
band = [transient_metrics(t_, probe(c, st["C_stray_F"])[1], ph, 1e-3)["droop"] for c in sample_fits(fit, 40, rng)]
check("sweep -> fit -> blind transient", abs(m_fit["level"] / m_true["level"] - 1) < 0.02
      and abs(m_fit["droop"] - m_true["droop"]) < 0.005 and min(band) - 0.003 <= m_true["droop"] <= max(band) + 0.003,
      f"plateau current {m_fit['level']*1e6:.2f} uA (true {m_true['level']*1e6:.2f}); droop {m_fit['droop']*100:.2f}% "
      f"(true {m_true['droop']*100:.2f}%, sampled fits {min(band)*100:.2f}-{max(band)*100:.2f}%); rise "
      f"{m_fit['rise_s']*1e6:.1f} us")
print("ALL PASS" if ok_all else "SOME FAILED")
sys.exit(0 if ok_all else 1)
