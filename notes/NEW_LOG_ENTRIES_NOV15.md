# Entries to add to validation_log.md (Nov 15, 2026: phantom becomes the primary validation)

Run each item on the Mac first; copy an entry only after your numbers match
(replace any that differ). Then delete this file.

2026-11-15 — DECISION (Hub): no reply from Dr. Muller, so the school phantom is
  the PRIMARY validation. provisional-prediction-v1 and runs/array_compare.py
  are kept untouched; prediction v2 is parked. Nothing deleted.

2026-11-15 — Phantom experiments beyond Parts 6-7 (src/phantom.py,
  runs/phantom_predict.py). Layouts in the setup file: E1 disk flush in the
  plate (12+ distances, full model vs point source), E2 the same wire end with
  no plate, E3 two more return positions (top sheet, ring). All predictions in
  mV per uA. DRAFT numbers on ASSUMED geometry (rerun with the measured setup):
  plate / no plate ratio 1.71 at 1 mm, 1.99 at 4 mm, 2.12 at 10 mm (not
  exactly 2: the tip sits above the plane, and the walls matter far away).
  Return ratio to the far-wall layout: 0.98 at 1 mm, 0.83 at 10 mm, 0.66 (top
  sheet) and 0.41 (ring) at 20 mm. Spread between the three returns: 2% at
  1 mm, 9% at 5 mm, 19% at 10 mm, 86% at 20 mm -- the array's return-path
  finding at phantom scale.
  tests/test_phantom_layouts.py: current conserved (1e-8) and RECIPROCITY on
  the solver in every layout (forward and swapped transfer resistance equal to
  6 digits, with plate, post, sheet or ring in place); A - B one number per
  layout; G identical at sigma 0.8 and 1.55 S/m (5e-9).

2026-11-15 — Every prediction is a band (95%): Monte Carlo (4000 samples) over
  probe position +-0.1 mm, tip height, electrode diameter and reference-wire
  position, with the grid uncertainty (GCI) drawn in the same samples;
  conductivity, temperature and current enter through V = I G / sigma.
  Draft: +-20% at 1 mm (position dominates), +-4% at 5-10 mm.

2026-11-15 — Pass criteria written BEFORE any measurement
  (notes/phantom_pass_criteria.md; resolved numbers in pass_criteria.json next
  to the predictions; to be committed under the phantom-prediction-v1 tag).
  Full model passes if >= k_min of n readings are covered (11 of 13). A simple
  model fails only at positions fixed in advance where it can be told apart,
  if the full model is closer in >= 75% of them and Wilcoxon p < 0.05.
  Draft: the fairest point source (half space, referenced) is distinguishable
  at 8 of 13 distances in E1.
  tests/test_phantom_compare.py: synthetic data from the full model pass; from
  a point-source world the point source is NOT failed; 30% errors rejected.

2026-11-15 — Interface: constant-phase element (src/transient.py,
  runs/interface_predict.py, runs/interface_compare.py). Fit R_s, R_ct, Q,
  alpha (+ stray capacitance from the lifted sweep), then predict the pulse
  transient blind. tests/test_transient.py: t^alpha charging law 0.07%;
  alpha = 1 matches an independent ODE solver (0.19%); fit recovers R_s 647
  (645), alpha 0.819 (0.82); CPE residual 0.014 vs ideal capacitor 0.248.
  Demo (made-up sweep): droop across a 200 us phase 4.06% [3.98, 4.14] vs
  3.37% for the ideal-capacitor fit. End-to-end test: traces made from the
  prediction pass; twice the droop does not.

2026-11-15 — Scope data loader (src/scope.py, runs/scope_plateau.py,
  runs/scope_noise.py). Reads plain, Rigol, Tektronix and Keysight/Siglent
  CSV layouts. Noise floor = random part + the quantization that averaging
  cannot remove. tests/test_scope.py: four layouts read identically; ADC step
  found from the data (62.50 uV); residual quantization formula vs simulation
  0.1023 vs 0.1022 step at noise 0.2 step; 64 traces: stated floor 4.21 uV vs
  scatter 4.16 uV. BUG FOUND by this test and fixed: a very quiet trace (fewer
  than 6 ADC levels) gave a wrong ADC step (2 mV instead of 62.5 uV) because
  no two neighbouring levels were present; the script now warns and asks for
  --vdiv.

2026-11-15 — Anisotropic block (src/aniso_phantom.py, runs/aniso_design.py,
  runs/aniso_predict.py). 1 mm plates, 1 mm gaps, staggered slots. Unit-cell
  simulation: 3 mm slots every 8 mm -> 2.85:1; 1 mm slots every 10 mm ->
  9.2:1 (converged within 1.3% between h = 1/5 and 1/9 mm). Staggered slots
  need far smaller pitch than aligned ones (aligned needs a 20 mm pitch for
  9:1 = 2 slot periods in the block).
  BUG FOUND and fixed before any number was used: the refined zones near the
  electrode ended at 0.5 and 1.5 mm, which are not grid nodes, so plate faces
  were off by up to 0.15 mm on the h = 1/3 mm grid. Zones now end at 1 and
  2 mm; solve_block refuses to run if any face is not midway between two
  nodes; tests/test_aniso_block.py checks it on four grids (offset 1e-14 mm).
  DRAFT (h = 1/3 mm only, assumed geometry): tensor model above the resolved
  model by 33% (3:1) / 71% (9:1) at 1 mm, 7% / 16% at 10 mm, 1.5% / 10% at
  15 mm, < 1% outside the block. Production grids (1/3 and 1/5 mm, 10 million
  nodes, 8-9 GB) not yet run: that is the Mac run that gets tagged.
  h = 1/7 mm (27 million nodes, ~24 GB) does not fit in 16 GB.

2026-11-15 — run_tests.py: 39 quick tests (was 33) + 6 slow.
