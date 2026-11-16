# Computation guide (research plan v16): what to run, in what order

Every script prints what it wrote. `--quick` versions are for testing only.
Times are for the MacBook (16 GB). Assumed numbers and their basis:
notes/ASSUMPTIONS.md.

## 0. One-time setup and checks
    xcode-select -p || xcode-select --install     # C++ compiler (Mac only, once)
    pip install "git+https://github.com/wagenadl/salpa@8c06ce7#subdirectory=python"
                                     # SALPA, the author's original code, built from his source at
                                     # commit 8c06ce7 (PyPI has no Mac build; on Linux/Windows
                                     # `pip install salpa` is the same code). Python >= 3.12.
                                     # Without it the SALPA arm is skipped with a message.
    python run_tests.py              # ~10 min, all should PASS (39 tests)
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

## 2. Phantom -- the PRIMARY validation (Nov 15). Predictions BEFORE Nov 21
Three separate blind predictions, each with its own git tag, each tagged before its data exist:
`phantom-prediction-v1` (field: Parts 6-7), `phantom-interface-prediction-v1` (pulse transient),
`aniso-prediction-v1` (anisotropic block, last).

### 2a. Before the first session
1. Copy `notes/phantom_setup_template.json` to `notes/phantom_setup.json` and replace every ASSUMED
   value with the measured one:
   - `instrument`: scope bits, smallest V/div, channels; generator channels and maximum amplitude;
     series resistor. (Not confirmed yet -- these only need editing here.)
   - `part6`: tank, fill depth, plate, stimulating disk, return wire, far reference wire, the 12+
     recording distances.
   - `experiments`: delete any layout that will not be built. E1 = disk flush in the plate,
     E2 = same wire end with no plate, E3_return_top / E3_return_ring = the other two returns.
   - `part7`: container, agar sphere, stimulating positions, return INSIDE the saline, traverse.
   - `uncertainties`: how well each input is known (these make the bands).
2. If the scope is available before tagging, measure the noise floor first (stimulus OFF, a few
   saved traces): `python runs/scope_noise.py data/phantom/raw/noise_*.csv --vdiv 0.002`
   and put its numbers in `instrument` and `part7.voltage_noise_V`.
3. Part 7 spacing (step 66): `python runs/traverse_design.py notes/phantom_setup.json`
4. `python runs/phantom_predict.py notes/phantom_setup.json --label phantom-prediction-v1`
   (~10 min, ~4.5 GB). Every prediction is a band in mV per uA. It also writes
   `pass_criteria.json`: the rules of `notes/phantom_pass_criteria.md` resolved for this geometry.
5. Commit and tag BEFORE any measurement (the pass-criteria file goes in the same commit):
       git add notes/phantom_setup.json notes/phantom_pass_criteria.md results/phantom_predictions figures/phantom_predictions
       git commit -m "Phantom blind prediction v1 + pass criteria"
       git tag phantom-prediction-v1 && git push && git push --tags

### 2b. At school
- Save every scope trace raw (never edit a raw file). One reading for the measurements CSV:
  `python runs/scope_plateau.py data/phantom/raw/<files for one position>.csv [--vdiv 0.002]`
  It prints plateau_V, plateau_V_sd, n_traces and noise_floor_V (8-bit quantization included).
- Fill copies of `notes/phantom_*_template.csv` in `data/phantom/`. Measurement rows have a
  `layout` column (E1_distance, ...) and a `condition` column (A = ground at the return,
  B = ground at the far reference wire). Two salines = two sessions with their own sigma.
- Compare: `python runs/phantom_compare.py results/phantom_predictions/phantom-prediction-v1 data/phantom/measurements.csv data/phantom/sessions.csv data/phantom/reciprocity.csv`
  It applies the committed criteria and prints the verdicts.

### 2c. Interface (constant-phase element), its own blind prediction
Order on the day: (1) impedance sweep, electrode in the cell and lifted out, with the wiring that
will be used for pulses; (2) predict, commit, tag; (3) only then record pulse traces.
    python runs/interface_predict.py data/phantom/impedance.csv notes/phantom_setup.json \
        results/phantom_predictions/phantom-prediction-v1 --sigma <measured> --label phantom-interface-prediction-v1
    git add results/interface_predictions figures/interface_predictions && git commit -m "Interface: blind pulse-transient prediction"
    git tag phantom-interface-prediction-v1 && git push && git push --tags
    python runs/interface_compare.py results/interface_predictions/phantom-interface-prediction-v1 --phase-us 200 data/phantom/raw/pulse_200us_*.csv
(`python runs/interface_predict.py --demo` shows the output on a made-up sweep.)

### 2d. Anisotropic block (experiment 8, built last), its own blind prediction
    python runs/aniso_design.py                      # ~2 min: which slots give which ratio
    python runs/aniso_predict.py notes/phantom_setup.json --label aniso-prediction-v1
                                                     # ~30 min, 8-9 GB: close every other program
Then commit and `git tag aniso-prediction-v1` before the block is measured. The block in the setup
file (`anisotropic`): 1 mm plates and gaps, slots 1 or 3 mm wide (odd), slot pitch an even number
of mm, staggered. Defaults: 3 mm slots every 8 mm (2.9:1) and 1 mm slots every 10 mm (9.2:1).

## 3. Aim 2 (Part 4)
| Plan step | Command | Output |
|---|---|---|
| 18 saturation map | `python runs/saturation_map.py` (~3 min); stimulator end of the band: `--rsrc 100000` | results/runs/saturation[_rsrc100k]/ |
| 19-31 budget study, 8 arms, variability sweep, spike-absorption check | `python runs/aim2_budget.py` (~2-3 h; leave it running). Uses the frozen `notes/aim2_split.json` -- never delete it | results/runs/aim2_budget/ |
| 33 latency classes | `python runs/aim2_latency_classes.py` | results/runs/aim2_budget/latency_classes.json |
| (Nov 10 exploration) | `python runs/aim2_recovery.py` | results/runs/aim2/ |
| 32 hybrid ground truth | after the phantom: recorded artifact traces + spikes added in software, 1-2 channels (design agreed with Hub first; to write once traces exist) | |

## 4. Aim 3 (Part 5)
| Plan step | Command | Output |
|---|---|---|
| 34-38 thresholds vs distance, membranes | `python runs/aim3_activation.py` (~30-60 min) | results/runs/aim3/ |
| 38-43 maps, sweeps, resolvability, Jaccard, Ronchi | `python runs/aim3_maps.py` (~45-60 min) | results/runs/aim3_maps/ |

## 5. Part 3 (PARKED Nov 15: only if Dr. Muller sends recordings; prediction v2 waits for his reply)
Convert his file to `data/array/areas.csv` (col,row,area_Vs,clipped), then
`python runs/array_compare.py results/runs/prediction-v2 data/array/areas.csv`.
Keep his raw files OUT of the public repo (data/array/ is git-ignored).

## Never
- overwrite a committed prediction (the scripts refuse; use a new --label)
- edit raw data in place
- change a parameter after looking at measurements without labelling the
  result "fitted" and keeping the blind one
- copy SALPA's source into this repository (it is GPL; it is installed with pip)
