# Entries to add to validation_log.md

Run each test on the Mac first. Copy an entry only after your own run matches;
replace any number that differs with yours. Then delete this file.
(Numbers below are from Claude's run of the same code, Nov 9, 2026.)

2026-11-09 — V5 grid convergence, 11.5 um square pad at 500 mV, sigma 1.5, quarter
  symmetry, outer faces grounded at 115 um. Three grids by halving (r = 2),
  pad spacing 0.72 / 0.36 / 0.18 um, up to 1.7M unknowns.
    delivered current     20.795 uA   p 1.10   extrapolated 20.556   GCI 1.44%
    phi above centre      156.45 mV   p 0.97   extrapolated 154.79   GCI 1.32%
    phi at 17.5 um        112.29 mV   p 0.97   extrapolated 110.69   GCI 1.77%
    phi at 35 um           46.65 mV   p 0.96   extrapolated  45.99   GCI 1.77%
  Finding: convergence is FIRST order (p ~ 1), not second. The current density is
  singular along the pad edge, and it limits the whole field.
  Cross-check: pad drawn with the edge row of nodes (too big by half a cell) and
  without it (too small) converge from opposite sides and extrapolate to the same
  current, 20.556 vs 20.551 uA (0.02%). PASS.
  Consequence: on the finest grid, array potentials carry ~1.8% discretization
  uncertainty; report Richardson-extrapolated values with the GCI.
  Also: V1/V2's 0.2% errors on the round disk were partly cancellation (the
  staircase disk is too big in some places and too small in others); the honest
  accuracy at that spacing is ~1-2%.

2026-11-09 — V6 reciprocity. Lopsided setup: pads of 11.5 and 8 um at unrelated
  positions, return = distant pad + top face, sigma 1.5 below 20 um and 0.5 above.
  Drive A read B = drive B read A = 4.730499 mV (difference 1.5e-8).
  Drive A read bare point P = drive P read A = 3.106144 mV (difference 1.5e-8). PASS.

2026-11-09 — Solver switched to classical (Ruge-Stuben) multigrid. On a 1.1M-unknown
  array problem: 35 CG iterations vs 266 for smoothed aggregation (7x faster).
  All earlier tests re-run: PASS, results unchanged (V5 identical; step 7 x1 0.245%,
  now 24 s instead of ~4 min). Memory ~1015-1075 B/unknown -> size check uses 1150.

2026-11-09 — MaxOne layout: 26,400 electrodes, 17.5 um pitch, 3.85 x 2.10 mm,
  MaxOne pad 8.75 x 12.00 um (105 um^2). Stimulating pad drawn with edges exactly
  on grid nodes for both chips at centre, corners and edge. PASS.

2026-11-09 — Interface circuit (series R + interface capacitance, biphasic pulse).
  Exact solution vs independent ODE solver: max difference 5e-9. C = 1/(2 pi f |Z|)
  reproduces MaxWell's injected-charge column (300 kOhm -> 0.106 nC, table 0.11).
  Phase charge exact; net charge zero. PASS.

2026-11-09 — Full array vs exact solution (top-plate return, image/Bessel series),
  coarse grid: 22,459 electrodes from 2 to 100 pitches: max error 0.79%,
  mean 0.22%. Current in = out. PASS.

2026-11-09 — Finding: the assumed periphery return ring (50 um wide) had GAPS on
  coarse grids when drawn by node position, so every grid had a different return
  path (60% disagreement at 1.75 mm). Fix: a node belongs to the ring if its cell
  touches it, plus smooth refinement to 12.5 um around the ring. The two finer
  grids then agree within 0.2% at every probe.

2026-11-09 — PROVISIONAL blind prediction (runs/predict_array.py), committed before
  any data. Settings: MaxOne, 1.5 S/m, 3.5 mm fluid in a 19 mm well, stim pad
  (110, 60), 500 mV, 200 us/phase. Three returns x three grids (up to 5.3M unknowns).
    pad-to-return resistance 27.4 kOhm (GCI 2.3%)
    potential per uA:  17.5 um 6.04-6.07 mV | 175 um 0.59-0.61 | 875 um 0.10-0.13
    return-path spread: 0.5% at 1 pitch, 4.8% at 10, 24.7% at 50, 52% at 100
    grid uncertainty < 1.3% within 20 pitches (all returns)
    interface: tau 7-15 us; charge/phase 133-265 pC at 500 mV;
    peak current 36.5 uA at the phase reversal (ideal source -> upper bound)
  Finding: near the pad the prediction does not depend on the unknown return path;
  far away it is dominated by it.
