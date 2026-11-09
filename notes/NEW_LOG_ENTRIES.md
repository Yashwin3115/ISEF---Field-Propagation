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
