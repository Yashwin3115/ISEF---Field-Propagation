"""Compare recorded pulse traces with the COMMITTED interface prediction.

    python runs/interface_compare.py results/interface_predictions/phantom-interface-prediction-v1 \\
        --phase-us 200 [--channel CH1] [--node-channel CH2] data/phantom/raw/pulse_200us_*.csv

The traces are averaged sample by sample (same trigger position), then the
same numbers are computed the same way as in the prediction
(src/transient.py, transient_metrics): plateau, droop, rise time; and, if a
second channel was clipped to the electrode side of the series resistor, the
node step and growth. Pass rule (notes/phantom_pass_criteria.md): the droop
lies inside its committed band widened by twice the measured droop's own
noise, and the rise time is no slower than the committed upper bound or the
scope's resolution (3 sample intervals), whichever is larger.
"""
import sys
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scope import read_scope_csv, average_traces
from transient import transient_metrics

def opt(name, default=None, cast=str):
    if name in sys.argv:
        k = sys.argv.index(name)
        v = cast(sys.argv[k + 1]); del sys.argv[k:k + 2]
        return v
    return default
phase_us = opt("--phase-us", None, float)
channel, node_channel = opt("--channel"), opt("--node-channel")
if len(sys.argv) < 3 or phase_us is None:
    sys.exit(__doc__)
pred = json.loads((Path(sys.argv[1]) / "prediction.json").read_text())
if pred.get("demo"):
    print("WARNING: this is a DEMO prediction, not a committed one.")
rec = pred["widths"][f"{phase_us:g}"]
ph = phase_us * 1e-6
data = [read_scope_csv(p) for p in sys.argv[2:]]
name = channel or next(iter(data[0]["channels"]))
t, v, sd, n = average_traces([(d["t"], d["channels"][name]) for d in data])
dt = float(np.median(np.diff(t)))
m = transient_metrics(t, v, ph)
# noise of the droop: two window means of the averaged trace
k = max(int(0.1 * ph / dt), 1)
base_sd = float(np.std(v[:max(len(v) // 20, 5)]))
u_droop = np.sqrt(2) * base_sd / np.sqrt(k) / abs(m["level"])
lo, hi = rec["droop"][0] - 2 * u_droop, rec["droop"][2] + 2 * u_droop
ok_d = lo <= m["droop"] <= hi
bound = max(rec["rise_s"][2], 3 * dt)
ok_r = m["rise_s"] <= bound
print(f"{n} trace(s) averaged, channel {name}, sample interval {dt*1e6:.3g} us")
print(f"plateau {m['level']*1e3:.4f} mV (predicted {rec['plateau_V'][1]*1e3:.4f} "
      f"[{rec['plateau_V'][0]*1e3:.4f}, {rec['plateau_V'][2]*1e3:.4f}])")
print(f"DROOP {m['droop']*100:.2f}% +- {2*u_droop*100:.2f}; predicted {rec['droop'][1]*100:.2f}% "
      f"[{rec['droop'][0]*100:.2f}, {rec['droop'][2]*100:.2f}]; ideal capacitor {rec['droop_ideal_capacitor']*100:.2f}% "
      f"-> {'INSIDE' if ok_d else 'OUTSIDE'}")
print(f"rise time {m['rise_s']*1e6:.2f} us; allowed up to {bound*1e6:.2f} us -> {'PASS' if ok_r else 'NOT PASSED'}")
out = {"n_traces": n, "measured": m, "droop_uncertainty": u_droop, "droop_pass": bool(ok_d), "rise_pass": bool(ok_r)}
if node_channel:
    _, vn, _, _ = average_traces([(d["t"], d["channels"][node_channel]) for d in data])
    te = m["t_edge_s"]
    step = float(np.interp(te + 0.02 * ph, t, vn) - np.interp(te - 0.02 * ph, t, vn))
    grow = float(np.interp(te + 0.9 * ph, t, vn) - np.interp(te + 0.1 * ph, t, vn))
    print(f"electrode node: step {step*1e3:.1f} mV (predicted {rec['node_step_V'][1]*1e3:.1f} "
          f"[{rec['node_step_V'][0]*1e3:.1f}, {rec['node_step_V'][2]*1e3:.1f}]); growth {grow*1e3:.1f} mV "
          f"(predicted {rec['node_growth_V'][1]*1e3:.1f} [{rec['node_growth_V'][0]*1e3:.1f}, "
          f"{rec['node_growth_V'][2]*1e3:.1f}])")
    out.update(node_step_V=step, node_growth_V=grow)
print(f"VERDICT: interface prediction {'PASSES' if ok_d and ok_r else 'DOES NOT PASS'}")
d = ROOT / "results" / "interface_comparison" / Path(sys.argv[1]).name
d.mkdir(parents=True, exist_ok=True)
(d / f"comparison_{phase_us:g}us.json").write_text(json.dumps(out, indent=2))
