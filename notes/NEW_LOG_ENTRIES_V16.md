# Entries to add to validation_log.md (research plan v16 work, Nov 11, 2026)

Run each item on the Mac first; copy an entry only after your numbers match
(replace any that differ). Then delete this file.

2026-11-11 — v16 update installed; run_tests.py: 33/33 PASS (fresh clone).
  New: phantom traverse/conditions, Aim 2 eight arms + variability sweep,
  original SALPA, pad-size and stimulator sweeps, assumptions note.

2026-11-11 — Phantom predictions rebuilt for v16 (runs/phantom_predict.py;
  DRAFT on ASSUMED geometry -- rerun with the measured setup and tag before
  Nov 21).
  Part 6 (steps 53-54): two referencing conditions. A - B = phi(reference) -
  phi(return) is ONE number at every position: G = 36.8 1/m (grid U 3.6%),
  1.19 mV at 50 uA; 89% of the condition-A reading at 20 mm.
  Grid uncertainty: A 1.8-3.2%, B 0.0-1.3%. Position (+-0.1 mm) 10% at 1 mm.
  Part 7 (steps 65-68): return inside the saline; traverse through the agar
  centre away from the stimulating electrode (12 points, 6 in agar, 6 in
  saline). Half-sphere symmetry checked against the full sphere (difference
  < 0.001). Tighter grid at the return: absolute values converge at order
  1.6 (U 2.7%). Gradient ratio from the 4 points nearest the interface
  (35, 37.5, 42.5, 45 mm): 0.256 (grid U 1.3%); ideal sigma_out/sigma_in
  0.276; gap -7%. At 0.5 mm spacing the solver gives 0.2749 vs 0.2759 ideal
  (the interface condition is honoured). Step 66 spacing: 2.5 mm (largest
  allowed by the 5 mm rule). Measurement uncertainty of the ratio: position
  8%, noise 32% at 5 uV per reading -> ~1 uV needed for a ~10% test.
  Analysis chain (tests/test_phantom_compare.py) PASS, incl. the exact
  two-layer profile (ratio 0.277778 = 1/3.6 exactly).

2026-11-11 — Aim 2, v16 (runs/aim2_budget.py, ~27 min here). Eight arms,
  10 draws x budgets 1-20, variability none/low/medium/high (ASSUMED levels).
  Recall of evoked spikes < 1 ms at <= 2 false/el/s, +-3 mV, 300 Hz high-pass:
                         none   low    medium  high
    physics per-pulse    0.96   0.96   0.95    0.91   (no calibration needed)
    physics fixed (b=20) 0.96   0.93   0.80    0.71
    low-rank per-pulse   0.83-0.90  0.81-0.91  0.79-0.90  0.71-0.81 (b=1..20)
    low-rank fixed       0.84-0.91  0.81-0.88  0.79      0.70-0.73
    SALPA (original)     0.86   0.84   0.83    0.73
    SALPA + forced peg   0.02   0.03   0.03    0.04
    per-channel average  0.44 -> 0.31 (falls with budget: absorbs evoked spikes)
    blanking             0.13   0.13   0.12    0.11
  Physics advantage, fixed vs fixed: +0.05-0.12 at none -> +0.00-0.01 at
  medium/high (practically gone; never <= 0 at every budget, so the plan's
  "disappears" rule is not met). Per-pulse vs per-pulse: +0.05-0.21 at every
  level. Earliest recovery: physics 0.25 ms (none-medium), 2.0 ms at high.
  Spike absorption (per-pulse physics): parameter shift 0.2-1.0% of value;
  0.40-0.55 x the pulse-to-pulse SD at none/low (where that SD is only
  fitting noise), 0.09-0.24 x at medium/high. No recall cost (0.96 = fixed).
  Stress test (12 wrong templates): periphery-ring templates fail (0.84), all
  attributed to the spatial pattern, all four "best explained by the well
  wall" (the true return). Others 0.96.
  Bug found and fixed before the final run: the calibration mean required a
  sample to be usable in EVERY calibration pulse, which left electrodes that
  clipped once uncorrected in the empirical templates (handicapped baseline).
  Now each sample is averaged over the pulses in which it was usable.

2026-11-11 — SALPA (step 28): the ORIGINAL implementation, used directly:
  PyPI `salpa` 1.0 (author D. A. Wagenaar; github.com/wagenadl/salpa commit
  8c06ce7; GPL-3.0-or-later), installed with pip, not copied. Author's
  command-line defaults (src/salpa.cpp): tau 3 ms, blank-depeg 20, ahead 5,
  asymmetry 10 samples, threshold 3 x noise SD. Snippets embedded in
  generated baseline noise (SALPA needs ~4 ms of history). Test: residual
  5.9 uV after a 2 mV slow artifact (noise 5). Its after-peg blanking (20
  samples = 1 ms at 20 kHz) hides the first millisecond.

2026-11-11 — Stimulator output resistance 0-100 kOhm (runs/stimulator_sweep.py).
  Charge per phase fixed at C V = 177 pC. Peak artifact at the neighbour:
  224 -> 47 mV instantaneous, 42 -> 29 mV as sampled (50 us). Band:
  1 pitch 47-224 mV, 10 pitches 4.6-22 mV, 100 pitches 0.32-1.5 mV.
  Saturation map at +-3 mV: 40-65% of the array (ideal source) vs 2.5-2.9%
  (100 kOhm). At +-1 mV: 100% vs 19-29%.

2026-11-11 — Pad size 8.75x12 vs 5x9 um (runs/pad_size_sensitivity.py).
  Transfer resistance: 0.7% at 1 pitch, identical beyond. Small pad:
  R_sol 26.9 -> 40.8 kOhm, peak current and artifact -34%; saturated
  fraction at +-3 mV 42% -> 21%. Aim 2: large-pad template fits the small-pad
  pattern within 0.4% -> no change. Aim 3 (|Z| 450 kOhm): over the pad
  163 -> 127 mV (same |Z|) or 227 mV (C scaled with area); at 10 um 428 ->
  519 or 936 mV. Locality conclusion unchanged (<= ~10 um at 500 mV).

2026-11-11 — notes/ASSUMPTIONS.md: every assumed number with its stated basis,
  incl. the 300 Hz high-pass (Ronchi et al. 2019 noise bands -- a stated
  basis, not confirmation of the lab's setting).
