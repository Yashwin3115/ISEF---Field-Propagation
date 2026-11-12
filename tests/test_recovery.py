"""Checks for src/recovery.py (Aim 2 tools), on small made-up arrays.

1. bin_average of a constant is the constant; of a ramp is the bin mid-value
2. the amplifier clips at +-sat and flags those samples (plus `recover` after)
3. a planted spike in pure noise is detected, at the right sample
4. fit_physics with the TRUE spatial pattern recovers the artifact to noise
   level and the right time constant
5. onset gains recover an arbitrary (non-solver) spatial pattern exactly;
   saturated electrodes get the fallback value
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recovery import (bin_average, record, detect, noise_level, spike_shape, fit_physics,
                      unit_current_fine, unit_current_binned, electrode_gains, TS)

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

rng = np.random.default_rng(7)
SUB = 20
T0, T1 = -1e-3, 3e-3
n = int(round((T1 - T0) / TS))
tf = T0 + (np.arange(n * SUB) + 0.5) * TS / SUB
ts = T0 + (np.arange(n) + 0.5) * TS

# 1
c = bin_average(np.full((2, 40), 3.0), 20)
r = bin_average(np.arange(40.0)[None, :], 20)
check("bin average", np.allclose(c, 3) and np.allclose(r, [[9.5, 29.5]]), f"{r.ravel()}")

# 2
x = np.zeros((1, n * SUB)); x[0, 30 * SUB:30 * SUB + 3] = 1.0
y, s = record(x, np.zeros_like(x), 0.0, 0.1, SUB, 2, rng)
check("clipping + recovery flag", np.max(y) <= 0.1 + 1e-12 and s[0, 30:33].all() and not s[0, 33]
      and s.sum() == 3, f"flagged samples {np.flatnonzero(s[0])}")

# 3
sp = np.zeros((1, n * SUB)); t_sp = 1.2e-3
sp[0] = 80e-6 * spike_shape((tf - t_sp) * 1e3)
y, s = record(np.zeros_like(sp), sp, 5e-6, 1.0, SUB, 0, rng)
sig = noise_level(y, ts < 0)
d = detect(y, ~s, sig)
want = int(np.floor((t_sp - T0) / TS))
check("spike detection", len(d) == 1 and abs(d[0][1] - want) <= 1, f"found {d}, expected sample {want}")

# 4 true pattern: template should match the truth
n_el = 50
Z = 1e3 / (1 + np.arange(n_el))                  # ohm
R, C = 25e3, 0.4e-9
art = 0.5 * Z[:, None] * unit_current_fine(R, C, tf - T0, -T0)[None, :]
y, s = record(art, np.zeros_like(art), 5e-6, 1.0, SUB, 0, rng)
win = (ts > -0.1e-3) & (ts < 1.5e-3)
model, coef, C_fit = fit_physics(y, ~s, Z, R, np.geomspace(0.1e-9, 1.5e-9, 41), tf - T0, -T0, SUB,
                                 window=win)
res = (y - model)[:, win]
check("physics fit, true pattern", np.std(res) < 6e-6 and abs(C_fit / C - 1) < 0.05,
      f"residual {np.std(res)*1e6:.2f} uV rms (noise 5), C {C_fit*1e9:.3f} nF (true 0.400)")

# 5 onset gains: arbitrary pattern, electrode 0 saturated at onset
G = rng.uniform(-1, 1, n_el) * 1e3
ib = unit_current_binned(R, C, tf - T0, -T0, SUB)
y = G[:, None] * ib[None, :]
usable = np.ones_like(y, dtype=bool); usable[0, ts < 0.2e-3] = False
onset = (ts > -0.1e-3) & (ts < 0.1e-3)
g, meas = electrode_gains(y, usable, ib, onset, np.full(n_el, -99.0))
check("onset gains", np.allclose(g[1:], G[1:], rtol=1e-9) and g[0] == -99.0 and not meas[0],
      f"max rel. error {np.max(np.abs(g[1:]/G[1:]-1)):.1e}; saturated electrode -> fallback")

# 6 SALPA, original implementation (skipped if the package is not installed)
from recovery import salpa_available, salpa_clean
if salpa_available():
    tt = -1e-3 + np.arange(120) * TS
    art = np.where(tt > 0.2e-3, 2e-3 * np.exp(-(tt - 0.2e-3) / 0.6e-3), 0)
    y = (art + rng.normal(0, 5e-6, 120))[None, :]
    u = np.ones_like(y, bool); u[0, 20:25] = False; y[0, 20:25] = 3e-3
    c, ok = salpa_clean(y, u, np.array([5e-6]), 3e-3)
    late = ok[0] & (tt > 2.5e-3)
    rms = np.std(c[0][late]) * 1e6
    check("SALPA (original) removes a slow artifact", late.sum() > 20 and rms < 8,
          f"residual {rms:.1f} uV rms after 2.5 ms (noise 5; artifact up to 2000 uV)")
else:
    print("SKIP  SALPA (original): package not installed -- pip install salpa (Python >= 3.12)")

print("ALL PASS" if ok_all else "SOME FAILED")
