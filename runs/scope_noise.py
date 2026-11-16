"""Noise floor of the scope + probe + tank, from traces recorded with the
stimulus OFF (plan step 58). Run this on Day 1, BEFORE the predictions are
tagged if possible: its numbers go in notes/phantom_setup.json
("instrument" block and part7.voltage_noise_V), so the committed bands and the
Part 7 design use the real noise.

    python runs/scope_noise.py data/phantom/raw/noise_*.csv [--channel CH1] [--vdiv 0.002]
                               [--bits 8] [--divs 8] [--phase-us 200] [--target-uV 1]

Reports: the ADC step, single-sample noise, how many samples are independent,
mains hum (50/60 Hz) if the record is long enough, and a table of the plateau
noise floor against the number of averaged traces -- including the part that
averaging cannot remove.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scope import (read_scope_csv, quantization, effective_samples, lsb_from_settings,
                   residual_quantization_error, traces_needed)

def opt(name, default=None, cast=float):
    if name in sys.argv:
        k = sys.argv.index(name)
        v = cast(sys.argv[k + 1]); del sys.argv[k:k + 2]
        return v
    return default
channel = opt("--channel", None, str)
vdiv, bits, divs = opt("--vdiv"), opt("--bits", 8, int), opt("--divs", 8.0)
phase_us, target = opt("--phase-us", 200.0), opt("--target-uV", 1.0) * 1e-6
files = sys.argv[1:]
if not files:
    sys.exit(__doc__)
V, raw, dts = [], [], []
for p in files:
    d = read_scope_csv(p)
    name = channel or next(iter(d["channels"]))
    v = d["channels"][name]; raw.append(v); V.append(v - np.mean(v)); dts.append(np.median(np.diff(d["t"])))
dt = float(np.median(dts))
allv = np.concatenate(V)
q = quantization(np.concatenate(raw))
lsb = lsb_from_settings(vdiv, bits, divs) if vdiv else q["lsb_V"]
if not vdiv and lsb and not q["lsb_reliable"]:
    print("WARNING: the traces use fewer than 6 ADC levels, so the ADC step cannot be read from the data. "
          "Rerun with --vdiv <volts per division>.")
sd = float(np.std(allv))
fr = float(np.median([effective_samples(v) / len(v) for v in V]))
print(f"{len(files)} trace(s), {len(V[0])} samples each, sample interval {dt*1e6:.3g} us")
print(f"single-sample noise {sd*1e6:.1f} uV rms; independent fraction of samples {fr:.2f}")
if lsb:
    print(f"ADC step {lsb*1e6:.1f} uV -> noise = {sd/lsb:.2f} steps; quantization left after averaging "
          f"{residual_quantization_error(lsb, sd)*1e6:.3f} uV per level")
else:
    print("samples are not on an ADC grid; pass --vdiv to include quantization")
T = len(V[0]) * dt
for f0 in (50.0, 60.0):
    if T >= 2 / f0:
        t = np.arange(len(V[0])) * dt
        amp = np.mean([2 * abs(np.mean(v * np.exp(-2j * np.pi * f0 * t))) for v in V])
        print(f"{f0:.0f} Hz component: {amp*1e6:.1f} uV amplitude" + ("  <- mains hum" if amp > 0.5 * sd else ""))
m = int(round(phase_us * 1e-6 * 0.5 / dt))
print(f"\nplateau window: middle half of a {phase_us:.0f} us phase = {m} samples")
if m < 1:
    sys.exit("the sample interval is longer than the plateau window: use a faster time/div")
print(f"{'traces averaged':>16} {'noise floor of one reading':>28}")
qv = np.sqrt(2) * residual_quantization_error(lsb, sd) if lsb else 0.0
per = sd * np.sqrt(2.0 / max(m * fr, 1))
for n in (1, 4, 16, 64, 128, 256, 512, 1024):
    print(f"{n:16d} {np.hypot(per/np.sqrt(n), qv)*1e6:25.2f} uV")
n_need, ok = traces_needed(sd, target, m, fr, lsb)
print(f"\nfor {target*1e6:.1f} uV per reading: " + (f"average {n_need} traces" if ok else
      f"NOT reachable by averaging (quantization alone leaves {qv*1e6:.2f} uV): use a smaller V/div, "
      "a longer pulse, a larger current, or measure the difference between two probes"))
print("Put the single-sample noise and the ADC step in the 'instrument' block of "
      "notes/phantom_setup.json, and the reachable noise floor in part7.voltage_noise_V.")
