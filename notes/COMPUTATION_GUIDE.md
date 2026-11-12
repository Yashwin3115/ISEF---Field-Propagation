# Computation guide (research plan v13): what to run, in what order

Every script prints what it wrote. `--quick` versions are for testing only.
Times are for the MacBook (16 GB).

## 0. Check everything works
    python run_tests.py              # ~6 min, 32 tests, all should PASS
    python run_tests.py --all        # + slow checks (~30 min more)

## 1. Aim 1 (Part 1-2): the solver and the array
| Plan step | Command | Output |
|---|---|---|
| 2 anisotropic verification | `python tests/test_anisotropic.py` | results/anisotropic_verification.json |
| 2 MaxOne at 9:1:1 | `python runs/anisotropic_maxone.py` (~3 min) | results/runs/anisotropic/ |
| 7-13 blind prediction v1 | already committed + tagged `provisional-prediction-v1` | results/predictions/ |
| 15 prediction v2 | after Dr. Muller replies: edit PROVISIONAL in src/maxone.py, then `python runs/predict_array.py --label prediction-v2`, commit, `git tag prediction-v2` BEFORE opening his recordings | results/runs/prediction-v2/ |

## 2. Phantom (Parts 6-7) -- BEFORE Nov 21
1. Copy `notes/phantom_setup_template.json` to `notes/phantom_setup.json`; fill in
   the real numbers (every field marked ASSUMED). Measure with calipers.
2. `python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1`  (~3 min, ~4.5 GB)
3. Check the printed table and figure, then commit and tag BEFORE measuring:
       git add notes/phantom_setup.json results/phantom_predictions figures/phantom_predictions
       git commit -m "Phantom blind prediction v1"
       git tag phantom-prediction-v1 && git push && git push --tags
4. At school: save every scope trace (raw, never edited). Fill
   `notes/phantom_*_template.csv` copies in `data/phantom/`.
   Plateau voltages: `python runs/scope_plateau.py data/phantom/raw/<files>.csv`
5. After the session: `python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 data/phantom/measurements.csv data/phantom/sessions.csv data/phantom/reciprocity.csv`
6. Impedance sweep (step 56): `python runs/impedance_fit.py data/phantom/impedance.csv 100000 results/phantom_predictions/phantom-prediction-v1 <sigma>`

## 3. Aim 2 (Part 4)
| Plan step | Command | Output |
|---|---|---|
| 18 saturation map | `python runs/saturation_map.py` (~3 min) | results/runs/saturation/ |
| 19-30 budget study | `python runs/aim2_budget.py` (~30-60 min). First run creates `notes/aim2_split.json` -- COMMIT it; never delete it | results/runs/aim2_budget/ |
| 32 latency classes | `python runs/aim2_latency_classes.py` | results/runs/aim2_budget/latency_classes.json |
| (Nov 10 exploration) | `python runs/aim2_recovery.py` | results/runs/aim2/ |
| 31 hybrid ground truth | after the phantom: phantom artifact traces + synthetic spikes (to write once traces exist) | |

## 4. Aim 3 (Part 5)
| Plan step | Command | Output |
|---|---|---|
| 33-39 thresholds vs distance, membranes | `python runs/aim3_activation.py` (~30-60 min) | results/runs/aim3/ |
| 37-42 maps, sweeps, resolvability, Jaccard, Ronchi | `python runs/aim3_maps.py` (~45-60 min) | results/runs/aim3_maps/ |

## 5. Part 3 (only if Dr. Muller sends recordings)
Convert his file to `data/array/areas.csv` (col,row,area_Vs,clipped), then
`python runs/array_compare.py results/runs/prediction-v2 data/array/areas.csv`.
Keep his raw files OUT of the public repo (plan section 4).

## Never
- overwrite a committed prediction (the scripts refuse; use a new --label)
- edit raw data in place
- change a parameter after looking at measurements without labelling the
  result "fitted" and keeping the blind one
