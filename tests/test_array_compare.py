"""Check runs/array_compare.py on a FAKE recording made from the committed v1
prediction: the well-wall pattern x a charge inside the committed range, 1%
noise, electrodes within 40 um flagged clipped. Pass if the script prefers
the well-wall return, its pattern R^2 > 0.999, and the clipped electrodes are
excluded. No real data involved."""
import subprocess, sys, json, shutil
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
pred = ROOT / "results" / "predictions"
lines = [l for l in (pred / "provisional_predictions.csv").read_text().splitlines() if not l.startswith("#")]
hdr = lines[0].split(","); rng = np.random.default_rng(0)
tmp = ROOT / "results" / "_test_array_compare"
tmp.mkdir(parents=True, exist_ok=True)
fake = tmp / "fake_areas.csv"
with open(fake, "w") as f:
    f.write("col,row,area_Vs,clipped\n")
    for l in lines[1::7]:
        v = dict(zip(hdr, l.split(",")))
        if float(v["distance_um"]) == 0:
            continue
        f.write(f"{v['col']},{v['row']},{float(v['Z_well_wall_ohm'])*1.6e-10*(1+rng.normal(0,0.01))},"
                f"{int(float(v['distance_um']) < 40)}\n")
out = subprocess.run([sys.executable, str(ROOT / "runs" / "array_compare.py"), str(pred), str(fake)],
                     capture_output=True, text=True, cwd=ROOT)
print(out.stdout[-900:])
rep = json.loads((ROOT / "results" / "array_comparison" / "predictions" / "comparison.json").read_text())
ok = (rep["preferred_return_path"] == "full model, well_wall"
      and rep["models"]["full model, well_wall"]["pattern_R2"] > 0.999 and "3 clipped" in out.stdout)
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree(ROOT / "results" / "array_comparison" / "predictions", ignore_errors=True)
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
