"""Anisotropic phantom, step 1: choose the slot pitch by simulating the
resolved unit cell (experiment 8).

    python runs/aniso_design.py            (~1 min)

Block: 1.0 mm insulating plates, 1.0 mm saline gaps (fixed -- see
src/aniso_phantom.py for why), slots 1 or 3 mm wide through the plates. The
slot width and pitch set the anisotropy. For each pitch this prints the effective conductivities
(as fractions of the saline's) and the ratio along / across the plates, for
ALIGNED slots and for STAGGERED slots, on three grids to show convergence,
and picks the slots nearest the targets 3:1 and 9:1.
"""
import sys
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from aniso_phantom import unit_cell_sigma_y, effective_tensor

QUICK = "--quick" in sys.argv
NS = (3, 5) if QUICK else (5, 9, 15)
PITCHES = (4, 6, 10) if QUICK else (2, 4, 6, 8, 10, 12, 14, 16, 20)
SLOTS = (1,) if QUICK else (1, 3)
out = {"targets": [3.0, 9.0], "plate_mm": 1.0, "gap_mm": 1.0, "rows": []}
print(f"{'slot mm':>7} {'pitch mm':>8} {'slots':>10} " + " ".join(f"{'sy/s h=1/'+str(n):>12}" for n in NS) +
      f" {'sx/s':>6} {'sz/s':>6} {'ratio x:y':>10}")
for stag in (False, True):
    for W in SLOTS:
        for P in PITCHES:
            if W >= P / 2:
                continue
            v = [float(unit_cell_sigma_y(P, slot_mm=W, staggered=stag, n=n)) for n in NS]
            sx, sz = 0.5, (1.0 + W / P) / 2.0
            r = {"pitch_mm": P, "slot_mm": W, "staggered": stag, "sigma_y_by_grid": v, "grid_n": list(NS),
                 "sigma_x": sx, "sigma_y": v[-1], "sigma_z": sz, "ratio_x_over_y": sx / v[-1],
                 "grid_change_last_two_rel": abs(v[-1] / v[-2] - 1)}
            out["rows"].append(r)
            print(f"{W:7d} {P:8d} {'staggered' if stag else 'aligned':>10} " + " ".join(f"{q:12.4f}" for q in v) +
                  f" {sx:6.3f} {sz:6.3f} {sx/v[-1]:10.2f}")
print()
for stag in (True, False):
    rows = [r for r in out["rows"] if r["staggered"] == stag]
    for tgt in out["targets"]:
        b = min(rows, key=lambda r: abs(np.log(r["ratio_x_over_y"] / tgt)))
        print(f"{'staggered' if stag else 'aligned':>9} slots, target {tgt:g}:1 -> {b['slot_mm']} mm slots every "
              f"{b['pitch_mm']} mm give {b['ratio_x_over_y']:.2f}:1  ({40 // b['pitch_mm']} slot periods across a "
              f"40 mm block; open fraction of each plate {b['slot_mm']/b['pitch_mm']*100:.0f}%)")
        out.setdefault("choice_staggered" if stag else "choice_aligned", {})[f"{tgt:g}"] = b
print("\nUse STAGGERED slots: the same ratio needs a much smaller pitch, so more slot periods fit in the "
      "block and it behaves more like a uniform medium. Put the chosen slots in the setup file under "
      "anisotropic.variants (slot_pitch_mm, slot_width_mm), then run runs/aniso_predict.py.")
d = ROOT / "results" / "runs" / ("aniso_design_quick" if QUICK else "aniso_design")
d.mkdir(parents=True, exist_ok=True)
(d / "design.json").write_text(json.dumps(out, indent=2))
print(f"wrote {d.relative_to(ROOT)}/design.json")
