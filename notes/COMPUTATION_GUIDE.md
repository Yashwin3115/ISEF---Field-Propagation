# Computation guide (research plan v16): what to run, in what order

Every script prints what it wrote. `--quick` versions are for testing only.
Times are for the MacBook (16 GB). Assumed numbers and their basis:
notes/ASSUMPTIONS.md.

## 0. One-time setup and checks
    pip install salpa                # SALPA, the author's original package (needs Python >= 3.12;
                                     # without it the SALPA arm is skipped with a message)
    python run_tests.py              # ~8 min, all should PASS
    python run_tests.py --all        # + slow checks (~35 min more)

## 1. Aim 1 (Parts 1-2): the solver and the array
| Plan step | Command | Output |
|---|---|---|
| 2 anisotropic verification | `python tests/test_anisotropic.py` | results/anisotropic_verification.json |
| 2 MaxOne at 9:1:1 | `python runs/anisotropic_maxone.py` (~3 min) | results/runs/anisotropic/ |
| 7-13 blind prediction v1 | already committed + tagged `provisional-prediction-v1` | results/predictions/ |
| pad size 8.75x12 vs 5x9 um | `python runs/pad_size_sensitivity.py` (~3 min) | results/runs/pad_size/ |
| stimulator output resistance 0-100 kOhm | `python runs/stimulator_sweep.py` (~2 min) | results/runs/stimulator_sweep/ |
| 15 prediction v2 | after Dr. Muller replies: edit PROVISIONAL in src/maxone.py, `python runs/predict_array.py --label prediction-v2`, commit, `git tag prediction-v2` BEFORE opening his recordings | results/runs/prediction-v2/ |

## 2. Phantom (Parts 6-7) -- predictions BEFORE Nov 21
1. Copy `notes/phantom_setup_template.json` to `notes/phantom_setup.json`; fill in the
   MEASURED numbers (every field marked ASSUMED): tank, fill depth, plate, return wire,
   far reference wire (Part 6, step 54), container, agar sphere and its offset, both
   stimulating positions, the Part 7 return INSIDE the saline (step 67), the traverse
   direction (through the agar centre, away from the stimulating electrode, step 68),
   nominal conductivities.
2. Choose the traverse points nearest the interface (step 66):
   `python runs/traverse_design.py notes/phantom_setup.json`
   Put the recommended distances in `traverse_distances_mm` (>= 4 in the agar, >= 4 in
   the saline, the rest within 5 mm of the interface).
3. `python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1`
   (~3 min, ~4.5 GB). Prints condition A, condition B and their difference (Part 6) and
   the gradient ratio with the ideal ratio and the gap (Part 7).
4. Commit and tag BEFORE any measurement:
       git add notes/phantom_setup.json results/phantom_predictions figures/phantom_predictions
       git commit -m "Phantom blind prediction v1"
       git tag phantom-prediction-v1 && git push && git push --tags
5. At school: save every scope trace raw. Fill copies of `notes/phantom_*_template.csv`
   in `data/phantom/` (Part 6 rows have a `condition` column: A = ground at the return,
   B = ground at the far reference wire). Plateau voltages:
   `python runs/scope_plateau.py data/phantom/raw/<files>.csv`
6. Compare: `python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 data/phantom/measurements.csv data/phantom/sessions.csv data/phantom/reciprocity.csv`
7. Impedance sweep (step 57): `python runs/impedance_fit.py data/phantom/impedance.csv 100000 results/phantom_predictions/phantom-prediction-v1 <sigma>`

## 3. Aim 2 (Part 4)
| Plan step | Command | Output |
|---|---|---|
| 18 saturation map | `python runs/saturation_map.py` (~3 min); stimulator end of the band: `--rsrc 100000` | results/runs/saturation[_rsrc100k]/ |
| 19-31 budget study, 8 arms, variability sweep, spike-absorption check | `python runs/aim2_budget.py` (~2-3 h; leave it running). Uses the frozen `notes/aim2_split.json` -- never delete it | results/runs/aim2_budget/ |
| 33 latency classes | `python runs/aim2_latency_classes.py` | results/runs/aim2_budget/latency_classes.json |
| (Nov 10 exploration) | `python runs/aim2_recovery.py` | results/runs/aim2/ |
| 32 hybrid ground truth | after the phantom: phantom artifact traces + synthetic spikes (to write once traces exist) | |

## 4. Aim 3 (Part 5)
| Plan step | Command | Output |
|---|---|---|
| 34-38 thresholds vs distance, membranes | `python runs/aim3_activation.py` (~30-60 min) | results/runs/aim3/ |
| 38-43 maps, sweeps, resolvability, Jaccard, Ronchi | `python runs/aim3_maps.py` (~45-60 min) | results/runs/aim3_maps/ |

## 5. Part 3 (only if Dr. Muller sends recordings)
Convert his file to `data/array/areas.csv` (col,row,area_Vs,clipped), then
`python runs/array_compare.py results/runs/prediction-v2 data/array/areas.csv`.
Keep his raw files OUT of the public repo (data/array/ is git-ignored).

## Never
- overwrite a committed prediction (the scripts refuse; use a new --label)
- edit raw data in place
- change a parameter after looking at measurements without labelling the
  result "fitted" and keeping the blind one
- copy SALPA's source into this repository (it is GPL; it is installed with pip)
