"""Run every verification test and print one summary table.

    python run_tests.py          all quick tests (~5 minutes)
    python run_tests.py --all    also the step 7 working-grid run and the quick
                                 Aim 3 runs (~25 more minutes)

Plan step 13: if any test fails, fix the solver and re-run ALL tests before
going on. Run this after every change to anything in src/.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

QUICK = [
    ("1D (a) uniform slab",          ["tests/test_a_uniform.py"]),
    ("1D (b) two-layer slab",        ["tests/test_b_two_layer.py"]),
    ("1D (c) manufactured",          ["tests/test_c_manufactured.py"]),
    ("1D (d) radial point source",   ["tests/test_d_point_source.py"]),
    ("2D manufactured",              ["tests/test_2d_manufactured.py"]),
    ("3D manufactured",              ["tests/test_3d_manufactured.py"]),
    ("3D stretched grid",            ["tests/test_nonuniform.py"]),
    ("3D insulating faces",          ["tests/test_3d_insulating.py"]),
    ("3D bad-input checks",          ["tests/test_input_checks.py"]),
    ("Step 7 point source (x2)",     ["tests/step7_point_source_cube.py", "2"]),
    ("Electrode toolkit checks",     ["tests/test_electrode_helpers.py"]),
    ("V1 disk, current mode",        ["tests/test_v1_disk_current.py"]),
    ("V2 spreading resistance",      ["tests/test_v2_spreading.py"]),
    ("V3 mode consistency",          ["tests/test_v3_mode_consistency.py"]),
    ("V4 current conservation",      ["tests/test_v4_conservation.py"]),
    ("V5 grid convergence + GCI",    ["tests/test_v5_grid_convergence.py"]),
    ("V6 reciprocity",               ["tests/test_v6_reciprocity.py"]),
    ("V7 half-space far field",      ["tests/test_v7_far_field.py"]),
    ("MaxOne layout and grid",       ["tests/test_maxone_layout.py"]),
    ("Interface circuit vs ODE",     ["tests/test_interface.py"]),
    ("Full array vs exact series",   ["tests/test_array_image_series.py"]),
    ("Prediction pipeline (quick)",  ["runs/predict_array.py", "--quick"]),
    ("Positions pipeline (quick)",   ["runs/positions_point_source.py", "--quick"]),
    ("Neuron models (HH, Mainen)",   ["tests/test_neuron.py"]),
    ("Aim 2 recovery tools",         ["tests/test_recovery.py"]),
    ("Aim 2 pipeline (quick)",       ["runs/aim2_recovery.py", "--quick"]),
    ("Anisotropic point source",     ["tests/test_anisotropic.py"]),
    ("Phantom analysis chain",       ["tests/test_phantom_compare.py"]),
    ("Array comparison (fake data)", ["tests/test_array_compare.py"]),
    ("Phantom predictions (quick)",  ["runs/phantom_predict.py", "--quick"]),
    ("Saturation map (quick)",       ["runs/saturation_map.py", "--quick"]),
    ("Aim 2 budget study (quick)",   ["runs/aim2_budget.py", "--quick"]),
    ("Stimulator sweep (quick)",     ["runs/stimulator_sweep.py", "--quick"]),
]
SLOW = [
    ("Step 7 point source (x1)",     ["tests/step7_point_source_cube.py"]),
    ("Aim 3 pipeline (quick)",       ["runs/aim3_activation.py", "--quick"]),
    ("Aim 3 maps (quick)",           ["runs/aim3_maps.py", "--quick"]),
    ("Anisotropic MaxOne (quick)",   ["runs/anisotropic_maxone.py", "--quick"]),
    ("Pad-size sensitivity (quick)", ["runs/pad_size_sensitivity.py", "--quick"]),
]

tests = QUICK + (SLOW if "--all" in sys.argv else [])
env = {"MPLBACKEND": "Agg"}          # save figures without opening windows

results = []
for name, cmd in tests:
    print(f"running {name} ...", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable] + cmd, cwd=ROOT, capture_output=True, text=True,
                          env={**__import__("os").environ, **env})
    dt = time.perf_counter() - t0
    out = proc.stdout + proc.stderr
    is_test = cmd[0].startswith("tests/")
    ok = proc.returncode == 0 and (("PASS" in out and "FAIL" not in out) if is_test else True)
    results.append((name, ok, dt))
    if not ok:
        print("\n".join(out.strip().splitlines()[-15:]))      # show what went wrong

print(f"\n{'test':<30} {'result':<7} {'time':>7}")
print("-" * 46)
for name, ok, dt in results:
    print(f"{name:<30} {'PASS' if ok else 'FAIL':<7} {dt:6.1f}s")
n_fail = sum(not ok for _, ok, _ in results)
print("-" * 46)
print("ALL PASS" if n_fail == 0 else f"{n_fail} FAILED -- fix before going on (plan step 13)")
sys.exit(1 if n_fail else 0)
