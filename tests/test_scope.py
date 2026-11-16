"""Checks for the scope data loader (src/scope.py), on made-up files.

1. the same trace written in four export layouts (plain, Rigol, Tektronix,
   Keysight/Siglent block) is read back identically
2. the ADC step is recovered from the data; averaged (off-grid) data is
   recognised as such
3. residual quantization error: lsb / sqrt(12) with no noise, negligible once
   the noise is half a step; the formula matches a direct simulation
4. plateau reading from 64 quantized traces: the mean is right within the
   stated noise floor, and the floor matches the scatter of repeated readings
5. with noise far below one step, averaging does NOT help, and the stated
   noise floor says so
6. traces_needed: reachable and unreachable targets
7. the two command-line scripts run on these files
"""
import sys
import subprocess
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scope import (read_scope_csv, quantization, lsb_from_settings, residual_quantization_error,
                   average_traces, plateau_reading, traces_needed)

ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

rng = np.random.default_rng(11)
LSB = lsb_from_settings(0.002, 8, 8)                    # 2 mV/div, 8 bits over 8 divisions = 62.5 uV
DT, N = 1e-6, 1000
t = (np.arange(N) - 200) * DT                           # trigger at sample 200

def pulse(amp, noise, rng):
    """Biphasic 200 us/phase pulse of plateau `amp`, with noise, quantized."""
    v = np.where((t >= 0) & (t < 200e-6), amp, 0.0) - np.where((t >= 200e-6) & (t < 400e-6), amp, 0.0)
    return np.round((v + rng.normal(0, noise, N)) / LSB) * LSB

tmp = Path(tempfile.mkdtemp())
v1 = pulse(2.0e-3, 150e-6, rng); v2 = pulse(1.0e-3, 150e-6, rng)

# 1 -- four layouts
(tmp / "plain.csv").write_text("Time (s),CH1,CH2\n" + "\n".join(f"{a:.9e},{b:.9e},{c:.9e}" for a, b, c in zip(t, v1, v2)))
(tmp / "rigol.csv").write_text("X,CH1,CH2,Start,Increment,\nSequence,Volt,Volt,,,\n"
                               + f"0,{v1[0]:.9e},{v2[0]:.9e},{t[0]:.9e},{DT:.9e},\n"
                               + "\n".join(f"{k},{b:.9e},{c:.9e},,," for k, (b, c) in enumerate(zip(v1[1:], v2[1:]), 1)))
meta = [("Record Length", "1000"), ("Sample Interval", f"{DT:.3e}"), ("Trigger Point", "200"), ("Source", "CH1"),
        ("Vertical Units", "V"), ("Vertical Scale", "2.0e-3"), ("Horizontal Units", "s")]
rows = []
for k in range(N):
    a, b = meta[k] if k < len(meta) else ("", "")
    rows.append(f"{a},{b},,{t[k]:.9e},{v1[k]:.9e},")
(tmp / "tek.csv").write_text("\n".join(rows))
(tmp / "block.csv").write_text("Record Length,1000\nSample Interval," + f"{DT:.3e}" + "\nVertical Scale,0.002\n"
                               "Model,XYZ\nSource,CH1,CH2\nSecond,Volt,Volt\n"
                               + "\n".join(f"{a:.9e},{b:.9e},{c:.9e}" for a, b, c in zip(t, v1, v2)))
det = []
good = True
for name, nch in (("plain", 2), ("rigol", 2), ("tek", 1), ("block", 2)):
    d = read_scope_csv(tmp / f"{name}.csv")
    ch = list(d["channels"].values())
    good &= len(ch) == nch and np.allclose(d["t"], t, atol=1e-12) and np.allclose(ch[0], v1, atol=1e-12)
    if nch == 2:
        good &= np.allclose(ch[1], v2, atol=1e-12)
    det.append(f"{name}->{d['format']} {list(d['channels'])}")
check("four export layouts read identically", good, "; ".join(det))

# 2 -- ADC step from the data
q = quantization(v1)
avg = np.mean([pulse(2.0e-3, 150e-6, rng) for _ in range(64)], axis=0) + rng.normal(0, 1e-9, N)
q2 = quantization(avg)
check("ADC step found from the data", q["lsb_V"] is not None and abs(q["lsb_V"] / LSB - 1) < 1e-6 and q2["lsb_V"] is None,
      f"{q['lsb_V']*1e6:.2f} uV (true {LSB*1e6:.2f}); {q['levels_used']} levels; averaged data -> not on a grid")

# 3 -- residual quantization error
r0 = residual_quantization_error(LSB, 0.0); r5 = residual_quantization_error(LSB, 0.5 * LSB)
s = 0.2 * LSB
true = np.linspace(0, LSB, 201)[:-1]
sim = np.array([np.mean(np.round((x + rng.normal(0, s, 40000)) / LSB) * LSB) - x for x in true])
rs = residual_quantization_error(LSB, s)
check("residual quantization error", abs(r0 / (LSB / np.sqrt(12)) - 1) < 0.01 and r5 < 0.002 * LSB
      and abs(np.sqrt(np.mean(sim ** 2)) / rs - 1) < 0.05,
      f"no noise {r0/LSB:.4f} step (1/sqrt12 = 0.2887); noise 0.5 step -> {r5/LSB:.1e} step; "
      f"noise 0.2 step: formula {rs/LSB:.4f}, simulated {np.sqrt(np.mean(sim**2))/LSB:.4f}")

# 4 -- plateau reading, noise above one step
AMP = 2.0173e-3
reads = [plateau_reading([(t, pulse(AMP, 150e-6, rng)) for _ in range(64)]) for _ in range(40)]
vals = np.array([r["plateau_V"] for r in reads]); floor = np.mean([r["noise_floor_V"] for r in reads])
check("plateau from 64 quantized traces", abs(vals.mean() - AMP) < 3 * floor / np.sqrt(40) + 1e-7
      and 0.6 < vals.std(ddof=1) / floor < 1.6 and reads[0]["values_on_adc_grid"],
      f"mean {vals.mean()*1e3:.4f} mV (true {AMP*1e3:.4f}); scatter of 40 readings {vals.std(ddof=1)*1e6:.2f} uV, "
      f"stated noise floor {floor*1e6:.2f} uV; noise/step {reads[0]['noise_over_lsb']:.1f}")

# 5 -- noise far below one step: averaging cannot help
qtr = [(t, pulse(AMP, 0.05 * LSB, rng)) for _ in range(64)]
quiet = plateau_reading(qtr, lsb=LSB)
blind = plateau_reading(qtr)                             # no V/div given: the step cannot be found from 3 levels
err = abs(quiet["plateau_V"] - AMP)
check("quiet signal: quantization dominates and is reported", quiet["quantization_V"] > 5 * quiet["standard_error_V"]
      and err < 3 * quiet["noise_floor_V"] and err > 3 * quiet["standard_error_V"]
      and "too few levels" in blind["lsb_from"] and quiet["lsb_from"] == "given",
      f"error {err*1e6:.1f} uV; random part {quiet['standard_error_V']*1e6:.2f} uV, quantization part "
      f"{quiet['quantization_V']*1e6:.1f} uV -> floor {quiet['noise_floor_V']*1e6:.1f} uV")

# 6 -- traces needed
n1, ok1 = traces_needed(150e-6, 1e-6, 100, 1.0, LSB)
n2, ok2 = traces_needed(0.05 * LSB, 1e-6, 100, 1.0, LSB)
check("traces needed for a target noise floor", ok1 and n1 == int(np.ceil((150e-6 * np.sqrt(2 / 100)) ** 2 / 1e-12))
      and not ok2, f"150 uV noise, 100 samples, 1 uV target -> {n1} traces; noise 0.05 step -> not reachable")

# 7 -- command-line scripts
files = []
for k in range(8):
    p = tmp / f"pos_{k}.csv"
    p.write_text("Time (s),CH1\n" + "\n".join(f"{a:.9e},{b:.9e}" for a, b in zip(t, pulse(AMP, 150e-6, rng))))
    files.append(str(p))
noise_files = []
for k in range(4):
    p = tmp / f"noise_{k}.csv"
    p.write_text("Time (s),CH1\n" + "\n".join(f"{a:.9e},{b:.9e}" for a, b in
                                              zip(t, np.round(rng.normal(0, 150e-6, N) / LSB) * LSB)))
    noise_files.append(str(p))
r1 = subprocess.run([sys.executable, str(ROOT / "runs" / "scope_plateau.py")] + files, capture_output=True, text=True)
r2 = subprocess.run([sys.executable, str(ROOT / "runs" / "scope_noise.py")] + noise_files, capture_output=True, text=True)
check("scope_plateau.py and scope_noise.py run", r1.returncode == 0 and r2.returncode == 0 and "noise_floor" in r1.stdout,
      (r1.stderr + r2.stderr).strip()[-200:] or "both exit 0")

print("ALL PASS" if ok_all else "SOME FAILED")
