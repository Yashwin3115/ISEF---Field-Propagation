"""Check the interface circuit (src/interface.py) against an independent
numerical ODE solver, with and without a charge-transfer (leak) resistor.

Also checks:
  - C_dl from impedance reproduces MaxWell's "injected charge at 200 mV"
    column (Electrical Stimulation Guide, Table 1): 0.05-0.11 nC for
    300-600 kOhm platinum black
  - phase-1 charge equals C * A * (1 - exp(-T / tau)) for a pure capacitor
  - a purely capacitive interface leaves zero net charge after the pulse
"""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interface import (capacitance_from_impedance, biphasic, circuit_current,
                       pulse_summary)

ok = []

# 1. MaxWell table
for z, q_table in [(300e3, 0.11e-9), (600e3, 0.05e-9), (70e3, 0.45e-9), (150e3, 0.21e-9)]:
    q = capacitance_from_impedance(z) * 0.2
    good = abs(q / q_table - 1) < 0.07          # table rounds to 2 digits
    ok.append(good)
    print(f"|Z| {z/1e3:5.0f} kOhm -> C {capacitance_from_impedance(z)*1e9:.3f} nF "
          f"-> q(200 mV) {q*1e9:.3f} nC   table {q_table*1e9:.2f} nC")

# 2. against an ODE solver
A, T = 0.5, 200e-6
R_sol, C = 26.8e3, capacitance_from_impedance(450e3)
for R_buf, R_ct in [(0.0, np.inf), (10e3, np.inf), (0.0, 2e6)]:
    steps = biphasic(A, T)
    def V(t):
        v = 0.0
        for tk, vk in steps:
            if t >= tk:
                v = vk
        return v
    R = R_buf + R_sol
    def rhs(t, vc):
        i = (V(t) - vc[0]) / R
        leak = 0.0 if np.isinf(R_ct) else vc[0] / R_ct
        return [(i - leak) / C]
    t_eval = np.linspace(0, 3 * T, 3001)[1:]
    sol = solve_ivp(rhs, (0, 3 * T), [0.0], t_eval=t_eval, max_step=0.2e-6,
                    rtol=1e-10, atol=1e-14)
    i_ode = np.array([(V(tt) - vc) / R for tt, vc in zip(sol.t, sol.y[0])])
    i_exact = circuit_current(sol.t, steps, R_sol, C, R_buf, R_ct)
    # compare away from the step instants (the ODE solver can't resolve a jump exactly)
    away = np.min(np.abs(sol.t[:, None] - np.array([0, T, 2 * T])[None, :]), axis=1) > 1e-6
    err = np.max(np.abs(i_ode[away] - i_exact[away])) / np.max(np.abs(i_exact))
    ok.append(err < 1e-5)
    print(f"R_buf {R_buf/1e3:4.0f} kOhm, R_ct {R_ct:>8.3g}: max difference vs ODE {err:.1e}")

# 3. charge checks, pure capacitor
s = pulse_summary(A, T, R_sol, C)
q1_exact = C * A * (1 - np.exp(-T / s["tau_s"]))
e1 = abs(s["charge_phase1_C"] / q1_exact - 1)
ok += [e1 < 1e-3, abs(s["net_charge_C"]) < 1e-3 * abs(s["charge_phase1_C"])]
# charges also checked by brute-force integration of i(t) on a very fine time grid
tt = np.linspace(0, T, 400001)
q1_num = np.trapezoid(circuit_current(tt, biphasic(A, T), R_sol, C), tt)
e2 = abs(q1_num / s["charge_phase1_C"] - 1)
ok.append(e2 < 1e-3)
print(f"\ntau {s['tau_s']*1e6:.2f} us, peak {s['peak_current_A']*1e6:.2f} uA "
      f"(= 2A/R at the phase reversal: {2*A/R_sol*1e6:.2f})")
print(f"phase-1 charge {s['charge_phase1_C']*1e12:.3f} pC  formula {q1_exact*1e12:.3f}  "
      f"brute-force {q1_num*1e12:.3f}")
print(f"net charge after the pulse {s['net_charge_C']*1e12:.2e} pC")

print("PASS" if all(ok) else "FAIL")
