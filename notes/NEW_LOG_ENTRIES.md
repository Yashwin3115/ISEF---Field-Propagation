# Entries to add to validation_log.md

Run each step on the Mac first. Copy an entry only after your own run matches;
replace any number that differs with yours. Then delete this file.
(Numbers below are from Claude's run of the same code, Nov 10, 2026.)

2026-11-10 — Grid: where the well wall or fluid surface arrives while cells are
  still growing, the last cells now land exactly on the wall by lowering the
  growth rate slightly (bisection), and refinement zones have smooth edges.
  Before: one 1.51x jump at the fluid surface on the middle grid. Now every
  neighbouring-cell ratio <= the set growth on all grids and positions
  (checked on 4,000 random axes too). Layout test now checks x, y and z strictly.
  Note: the v1 prediction (tag provisional-prediction-v1) used the old grid and
  stays as committed; new runs go to results/runs/<label>/ and cannot overwrite it.

2026-11-10 — Spec 2.6, floating-electrode check (runs/floating_pads.py).
  Provisional settings, 1 uA into pad (110, 60), the 8 surrounding pads resolved
  exactly (pad edges on grid nodes), 4.2M unknowns.
  A = pads ignored (production model); B = pads as floating metal, zero net current.
    nearest pads (17.5 um): B vs A -4.0% (x neighbour), +1.8% (y neighbour);
    diagonal (24.7 um): -0.1%
    5-100 pitches: <= 0.08%
    pad footprint average vs centre value: up to +3.8% (y neighbour)
    stimulating pad resistance: -2.8% with metal neighbours
  Finding: ignoring the recording pads is safe beyond the first ring (< 0.1%);
  the 8 nearest electrodes carry about +-4% model uncertainty.

2026-11-10 — Three stimulation positions x three return paths, finest grid
  (runs/positions_point_source.py): centre (110, 60), off-centre (55, 30),
  near edge (2, 60). Up to 5.6M unknowns, 8 minutes total.
  Pad-to-return resistance 26.74-26.94 kOhm in every case.
  Top plate / well wall: the three positions give the same fall-off (within
  0.06% toward vs away from the edge) -- the array edge is not a boundary for
  the current, because the fluid continues past it.
  Periphery ring: near-edge pad reads 2.1% lower toward the edge at 2 pitches,
  and up to ~50% lower far away -- the edge matters only if the return is there.

2026-11-10 — Model version 1 (ideal point source I/(2 pi sigma r)) vs version 2
  (full model), every electrode, centre pad, median error by distance:
                     1-2 p   2-5 p   5-10 p  10-20 p  20-50 p  50-100 p  100-250 p
    top plate        -0.5%   +1.2%   +2.9%   +5.9%    +15.1%   +32.7%    +55.4%
    well wall        -1.2%   -0.5%   -0.9%   -1.9%     -4.7%    -8.1%    -10.7%
    periphery        -0.9%   +0.2%   +0.5%   +1.1%     +3.4%   +12.4%    +34.4%
  Finding: the point-source formula is within ~1% out to 5 pitches (90 um) for
  every return path; beyond ~10 pitches its error depends on the return path and
  grows to tens of percent. Near the array edge with a periphery return it is off
  by 3-11% even at 1-5 pitches.
