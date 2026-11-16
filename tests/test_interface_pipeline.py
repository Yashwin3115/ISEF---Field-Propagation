"""End-to-end check of the blind interface (pulse-transient) pipeline on
made-up data: runs/interface_predict.py --demo, then runs/interface_compare.py.

1. the demo prediction runs and writes a band for every pulse width
2. traces generated FROM the predicted probe trace (plus scope noise and 8-bit
   quantization, 64 traces) pass the committed rule
3. traces whose plateau droops twice as much do NOT pass
"""
import sys
import json
import subprocess
import tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ok_all = True
def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")

r = subprocess.run([sys.executable, str(ROOT / "runs" / "interface_predict.py"), "--demo"], capture_output=True, text=True)
d = ROOT / "results" / "interface_predictions" / "interface_quick"
pred = json.loads((d / "prediction.json").read_text()) if r.returncode == 0 else {}
widths = list(pred.get("widths", {}))
check("demo prediction runs", r.returncode == 0 and len(widths) >= 1 and all(
      pred["widths"][w]["droop"][0] <= pred["widths"][w]["droop"][1] <= pred["widths"][w]["droop"][2] for w in widths),
      f"pulse widths {widths} us; droop bands " + ", ".join(
          f"{pred['widths'][w]['droop'][1]*100:.2f}%" for w in widths) if widths else r.stderr[-300:])

if widths:
    w = "200" if "200" in widths else widths[0]
    tr = np.genfromtxt(d / f"probe_trace_{w}us.csv", delimiter=",", names=True)
    t, v = tr["time_from_edge_s"], tr["V_probe_V"]
    ph = float(w) * 1e-6
    lsb = 0.002 * 8 / 256
    rng = np.random.default_rng(5)
    tmp = Path(tempfile.mkdtemp())
    def write(tag, vv):
        files = []
        for k in range(64):
            p = tmp / f"{tag}_{k}.csv"
            q = np.round((vv + rng.normal(0, 150e-6, len(vv))) / lsb) * lsb
            p.write_text("Time (s),CH1\n" + "\n".join(f"{a:.9e},{b:.9e}" for a, b in zip(t, q)))
            files.append(str(p))
        return files
    def compare(files):
        c = subprocess.run([sys.executable, str(ROOT / "runs" / "interface_compare.py"), str(d), "--phase-us", w] + files,
                           capture_output=True, text=True)
        line = [l for l in c.stdout.splitlines() if l.startswith("DROOP")]
        return c, (line[0] if line else c.stderr[-300:])
    c1, l1 = compare(write("good", v))
    check("traces made from the prediction pass", c1.returncode == 0 and "VERDICT: interface prediction PASSES" in c1.stdout, l1)
    # twice the droop: tilt the first phase about its middle
    droop = pred["widths"][w]["droop"][1]
    level = np.interp(0.5 * ph, t, v)
    in_phase = (t > 0) & (t < ph)
    v2 = v.copy(); v2[in_phase] -= level * droop * ((t[in_phase] - 0.5 * ph) / (0.8 * ph))
    c2, l2 = compare(write("bad", v2))
    check("twice the droop does not pass", c2.returncode == 0 and "DOES NOT PASS" in c2.stdout, l2)

print("ALL PASS" if ok_all else "SOME FAILED")
