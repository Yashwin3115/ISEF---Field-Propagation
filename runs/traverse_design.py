"""Plan v15 step 66: choose the spacing of the traverse points nearest the
agar-saline interface BEFORE the session, by error propagation.

    python runs/traverse_design.py [notes/phantom_setup.json]

Solves Part 7 at the medium grid with the setup's geometry and nominal
conductivities, then for spacings 0.5-6 mm (points at the interface -/+ d and
-/+ 2d) reports: the finite-difference ratio and its gap from the ideal ratio
sigma_out / sigma_in, the ratio uncertainty from +-0.1 mm position error and
from the voltage noise, and the combined value. Prints the recommended
traverse positions; put them in the setup file, then run phantom_predict.py.
The noise term uses `voltage_noise_V` from the setup (ASSUMED until the
session's noise floor is measured) and the LARGER planned current.
"""
import sys
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phantom import load_setup, solve_sphere, design_spacing, mm

setup = load_setup(sys.argv[1] if len(sys.argv) > 1 else None)
s7 = setup["part7"]
ratio_ideal = s7["sigma_out_nominal_S_per_m"] / s7["sigma_in_nominal_S_per_m"]
I = max(s7["current_nominal_A"])
out = {}
for i, off in enumerate(s7["stim_offsets_mm"]):
    r = solve_sphere(setup, i, 2.0, verbose=False)
    rows, best = design_spacing(r["G_traverse"], r["interface_m"], ratio_ideal,
                                s7["sigma_out_nominal_S_per_m"], I, s7["voltage_noise_V"],
                                s7["position_uncertainty_mm"] * mm)
    out[str(off)] = {"rows": rows, "best": best}
    print(f"\nstimulating electrode at {off} mm; ideal ratio sigma_out/sigma_in = {ratio_ideal:.3f}; "
          f"{I*1e6:.0f} uA, noise {s7['voltage_noise_V']*1e6:.1f} uV, position +-{s7['position_uncertainty_mm']} mm")
    print(f"{'d mm':>5} {'FD ratio':>9} {'gap':>7} {'U_pos':>7} {'U_noise':>8} {'combined':>9} "
          f"{'dV in':>8} {'dV out':>8}")
    for q in rows[::2]:
        print(f"{q['spacing_mm']:5.2f} {q['fd_ratio']:9.4f} {q['gap_rel']*100:6.1f}% {q['U_pos_rel']*100:6.1f}% "
              f"{q['U_noise_rel']*100:7.1f}% {q['combined_rel']*100:8.1f}% {q['dV_inside_pair_V']*1e6:6.1f}uV "
              f"{q['dV_outside_pair_V']*1e6:6.1f}uV")
    print(f"-> best spacing within the plan's 5 mm rule: {best['spacing_mm']:.2f} mm (combined "
          f"{best['combined_rel']*100:.1f}%); without the rule it would be "
          f"{best['best_without_5mm_rule_mm']:.2f} mm ({best['combined_without_5mm_rule_rel']*100:.1f}%)")
Ra = s7["agar_diameter_mm"] / 2
d = min(v["best"]["spacing_mm"] for v in out.values())
near = [Ra - 2 * d, Ra - d, Ra + d, Ra + 2 * d]
print(f"\nRECOMMENDED nearest points (distances from the agar centre, mm): "
      f"{', '.join(f'{v:.2f}' for v in near)}")
print("Keep >= 4 points inside the agar and >= 4 in the saline (plan step 68); update "
      "'traverse_distances_mm' in the setup file, then run runs/phantom_predict.py.")
(ROOT / "results" / "phantom_predictions").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "phantom_predictions" / "traverse_design.json").write_text(json.dumps(out, indent=2))
