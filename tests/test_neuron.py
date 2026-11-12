"""Checks on the Hodgkin-Huxley cable model (src/neuron.py).

1. Rest: with no input the membrane stays at -65 mV.
2. Single compartment: a 1 ms, 20 uA/cm^2 pulse fires an action potential and
   a 2 uA/cm^2 pulse does not; the waveform matches an independent ODE solver.
3. Conduction velocity of the squid giant axon (d = 476 um, R_a = 35.4 Ohm cm,
   18.5 C): Hodgkin & Huxley (1952) calculated 18.8 m/s.
5. Mainen-Sejnowski 37 C membrane (ModelDB 2488): has a stable rest (about
   -80 mV for a whole axon of initial-segment membrane; -70 mV in the
   published model, where the soma holds it there), a single compartment fires
   and matches an independent ODE solver, and a 1 um axon conducts at a speed
   typical of thin unmyelinated mammalian axons (well under 2 m/s).
4. Passive cable in an extracellular field: a point electrode 50 um from a
   1 um axon. In steady state the membrane potential must equal the exact
   cable-equation solution  V = G * (lambda^2 Ve''),  G(x) = exp(-|x|/lambda)/(2 lambda).
   This checks the activating-function coupling, the part that links the
   neuron to the field solver.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from neuron import Axon, HH, _rates, resting_gates, PassiveMembrane, MainenMembrane

ok = []

# 1. rest
ax = Axon(n=50, dx_cm=10e-4, d_cm=1e-4, dt_ms=0.002)
out = ax.run(10.0)
drift = np.max(np.abs(out["V"] - HH["V_rest"]))
ok.append(drift < 0.01)
print(f"1. rest: largest drift after 10 ms {drift:.2e} mV")

# 2. single compartment vs ODE solver
def single(I_amp, dt=0.001, T=8.0):
    a = Axon(n=1, dx_cm=1e-3, d_cm=1e-3, dt_ms=dt)
    r = a.run(T, I_inj_fn=lambda s: np.full((1, 1), I_amp if s * dt < 1.0 else 0.0),
              record=[0])
    return r["t_ms"], r["traces"][:, 0, 0]

t, v20 = single(20.0)
_, v2 = single(2.0)

def hh_ode(t, y, I_amp):
    V, n, m, h = y
    an, bn, am, bm, ah, bh = _rates(np.array(V))
    I = I_amp if t < 1.0 else 0.0
    dV = I - (HH["gNa"] * m**3 * h * (V - HH["ENa"]) + HH["gK"] * n**4 * (V - HH["EK"])
              + HH["gL"] * (V - HH["EL"]))
    return [dV, an * (1 - n) - bn * n, am * (1 - m) - bm * m, ah * (1 - h) - bh * h]

n0, m0, h0 = resting_gates()
sol = solve_ivp(hh_ode, (0, 8.0), [HH["V_rest"], n0, m0, h0], args=(20.0,), t_eval=t,
                max_step=0.005, rtol=1e-8, atol=1e-10)
peak_err = abs(v20.max() - sol.y[0].max())
time_err = abs(t[np.argmax(v20)] - t[np.argmax(sol.y[0])])
ok += [v20.max() > 30, v2.max() < -40, peak_err < 1.0, time_err < 0.05]
print(f"2. 20 uA/cm^2: peak {v20.max():+.1f} mV at {t[np.argmax(v20)]:.3f} ms "
      f"(ODE solver {sol.y[0].max():+.1f} mV at {t[np.argmax(sol.y[0])]:.3f} ms); "
      f"2 uA/cm^2: peak {v2.max():+.1f} mV (no spike)")

# 3. squid conduction velocity
def velocity(dx_cm, dt_ms):
    n = int(round(5.0 / dx_cm))
    a = Axon(n=n, dx_cm=dx_cm, d_cm=476e-4, Ra_ohm_cm=35.4, dt_ms=dt_ms, temp_C=18.5)
    i1, i2 = int(round(1.5 / dx_cm)), int(round(3.5 / dx_cm))
    def inj(s):
        I = np.zeros((n, 1))
        if s * dt_ms < 0.2:
            I[:3] = 2000.0
        return I
    r = a.run(4.0, I_inj_fn=inj, record=[i1, i2])
    tr = r["traces"][:, :, 0]
    t1 = r["t_ms"][np.argmax(tr[:, 0] > 0)]
    t2 = r["t_ms"][np.argmax(tr[:, 1] > 0)]
    return (2.0e-2) / ((t2 - t1) * 1e-3)            # m/s

v_a = velocity(0.02, 0.005)
v_b = velocity(0.01, 0.0025)
ok.append(abs(v_b / 18.8 - 1) < 0.05)
print(f"3. squid axon conduction velocity: {v_a:.2f} m/s (dx 200 um), {v_b:.2f} m/s "
      f"(dx 100 um); Hodgkin & Huxley 1952: 18.8 m/s")

# 4. passive cable in a point-source field, steady state
d, Ra, gL = 1e-4, 35.4, 0.3                       # cm, Ohm cm, mS/cm^2
lam = np.sqrt(d / (4 * Ra * gL * 1e-3))           # cm
h, sigma, I = 50e-4, 1.5e-2, 1e-6                 # cm, S/cm, A
dx = 2e-4
n = int(round(0.8 / dx)) | 1
x = (np.arange(n) - n // 2) * dx
Ve = I / (4 * np.pi * sigma * np.sqrt(x**2 + h**2)) * 1e3        # mV
ax = Axon(n=n, dx_cm=dx, d_cm=d, Ra_ohm_cm=Ra, dt_ms=0.02, membrane=PassiveMembrane(gL=gL))
r = ax.run(40.0, Ve_fn=lambda s: Ve[:, None])
V_sim = r["V"][:, 0] - HH["V_rest"]
# exact: V = integral G(x - s) lambda^2 Ve''(s) ds, on a fine grid
xs = np.linspace(-0.8, 0.8, 160001)
R2 = xs**2 + h**2
Vepp = I / (4 * np.pi * sigma) * (2 * xs**2 - h**2) / R2**2.5 * 1e3       # mV / cm^2
mid = slice(n // 2 - 200, n // 2 + 201)
V_ex = np.array([np.trapezoid(np.exp(-np.abs(xx - xs) / lam) / (2 * lam) * lam**2 * Vepp, xs)
                 for xx in x[mid]])
err = np.max(np.abs(V_sim[mid] - V_ex)) / np.max(np.abs(V_ex))
ok.append(err < 0.01)
print(f"4. passive cable, point electrode 50 um away: under the electrode simulated "
      f"{V_sim[n//2]:+.6f} mV, exact {V_ex[len(V_ex)//2]:+.6f} mV; max error {err*100:.4f}% "
      f"of peak (length constant {lam*1e4:.0f} um)")

# 5. Mainen-Sejnowski membrane
mem = MainenMembrane()
def single_m(I_amp, dt=0.001, T=4.0, dur=0.3):
    a = Axon(n=1, dx_cm=1e-3, d_cm=1e-3, dt_ms=dt, membrane=mem)
    r = a.run(T, I_inj_fn=lambda s: np.full((1, 1), I_amp if s * dt < dur else 0.0), record=[0])
    return r["t_ms"], r["traces"][:, 0, 0]

def ms_ode(t, y, I_amp, dur=0.3):
    V, m, h, n = y
    minf, mtau, hinf, htau = mem._na(np.array(V))
    ninf, ntau = mem._kv(np.array(V))
    I = I_amp if t < dur else 0.0
    Iion = (mem.gna * m**3 * h * (V - mem.ENa) + mem.gkv * n * (V - mem.EK)
            + mem.gl * (V - mem.El))
    return [(I - Iion) / mem.C, (minf - m) / mtau, (hinf - h) / htau, (ninf - n) / ntau]

tm, vm = single_m(400.0)
_, vm_low = single_m(1.0)
g0 = mem.init_gates(mem.V_rest)
sol = solve_ivp(ms_ode, (0, 4.0), [mem.V_rest, *[float(g) for g in g0]], args=(400.0,),
                t_eval=tm, method="LSODA", max_step=0.002, rtol=1e-8, atol=1e-10)
pk_err = abs(vm.max() - sol.y[0].max())
tk_err = abs(tm[np.argmax(vm)] - tm[np.argmax(sol.y[0])])
above = tm[vm > (vm.max() + mem.V_rest) / 2]
width = above[-1] - above[0]

ax37 = Axon(n=400, dx_cm=2e-4, d_cm=1e-4, Ra_ohm_cm=150.0, dt_ms=0.001, membrane=mem)
def inj37(s):
    I = np.zeros((400, 1))
    if s * 0.001 < 0.1:
        I[:5] = 5000.0
    return I
r37 = ax37.run(2.0, I_inj_fn=inj37, record=[100, 300])
tr = r37["traces"][:, :, 0]
arrived = bool((tr[:, 1] > 0).any() and (tr[:, 0] > 0).any())
v37 = (200 * 2e-4 * 1e-2) / ((r37["t_ms"][np.argmax(tr[:, 1] > 0)]
                              - r37["t_ms"][np.argmax(tr[:, 0] > 0)]) * 1e-3) if arrived else float("nan")
ok += [-85 < mem.V_rest < -65, vm.max() > 20, vm_low.max() < -40, pk_err < 2.0,
       tk_err < 0.02, arrived, 0.1 < v37 < 2.0]
print(f"5. Mainen-Sejnowski 37 C: rest {mem.V_rest:.2f} mV; single compartment peak "
      f"{vm.max():+.1f} mV at {tm[np.argmax(vm)]:.3f} ms (ODE solver {sol.y[0].max():+.1f} mV at "
      f"{tm[np.argmax(sol.y[0])]:.3f} ms), half-width {width:.3f} ms; weak input peak "
      f"{vm_low.max():+.1f} mV; 1 um axon conducts: {arrived} ({v37:.2f} m/s)")

print("PASS" if all(ok) else "FAIL")
