2026-10-24 — Test (a) uniform slab, N=20: max error 2.8e-16. PASS.

2026-10-24 — Test (b) two-layer slab, σ1=1, σ2=10, interface at midpoint between nodes.
  Harmonic averaging: error ~1e-15 at all N. PASS.
  Arithmetic averaging: error 2.9e-2 at N=21, first-order convergence (ratio ~2).
  Decision: use harmonic averaging for all conductivity jumps.

2026-10-24 — Test (c) manufactured solution, phi=sin(pi x), sigma=1+x, harmonic averaging.
  N=10→320: error 9.4e-3 → 9.1e-6, ratio 4.00, observed order 2.00. PASS.

2026-10-24 — Test (d) radial point source, I=50 uA, sigma=1.5 S/m, r0=0.25 mm, R=150 mm.
  Max rel. error over 0.5–5 mm: 8.9e-3 (N=600) → 2.0e-5 (N=19200).
  Ratio rises 2.7→3.9 (pre-asymptotic while h ~ r0), observed order 1.96. PASS.

2026-10-24 — 2D manufactured solution, phi=sin(pi x)e^y, sigma=1+x+y, hx≠hy.
  Matrix symmetric (|A-A^T|=0). N=10→320: error 2.9e-3 → 2.9e-6, order 2.00. PASS.

2026-10-24 — 3D manufactured solution, phi=sin(pi x)sin(pi y)e^z, sigma=1+x+y+z, CG+AMG, rtol 1e-8.
  Symmetric; CG vs direct ~1e-10. N=8→64 (250k unknowns): order 2.00, CG iters 6→10. PASS.

2026-10-25 — Step 4 nonuniform grid. Finite-volume form, symmetric matrix.
  Stretched-grid test (neighbor ratio up to 1.42), two halvings: ratio 3.97, 3.99, order 2.00. PASS.
  Tank grids: 13.1M / 7.9M / 4.3M unknowns (x1, x4/3, x2).
  MacBook memory: 872 B/unknown (1M), 756 B/unknown (4M) → using 900. Working grid ~11.8 GB of 16.

2026-10-25 — Solver allows insulating (zero-current) outer faces.
  Insulating-face manufactured test: order 2.00. PASS.
  Input checks added (bad sigma, NaN, shape mismatch, bad grid, no fixed nodes). PASS.

2026-10-25 — Step 7, point source in 300 mm cube, solved as one octant (symmetry planes).
  Full cube would need 22M unknowns (~20 GB); octant needs 2.7M.
  Max rel. error 0.5–5 mm: x2 0.90%, x4/3 0.42%, x1 0.25%. PASS (< 2%).
  Step 10 check: x1 vs x4/3 differ by 0.18% (< 1%). PASS.
  Finding: coarsening h_min and h_max alone leaves the 10%-growth zone unchanged;
  growth rate must be scaled too (1.10 → 1.133 → 1.20).

2026-11-08 — Solver core rebuilt for the MaxOne array problem.
  One conductance matrix over all nodes; (L phi)_i = current out of node i.
  Node types: fixed (voltage-mode electrode, return contact), floating (current-mode
  electrode, passive recording electrode), free. Unfixed outer faces = insulating
  (substrate at z = 0). Delivered current computed from L, not assumed.
  All earlier tests re-run: PASS. Step 7 results unchanged (x2 0.898%, x1 0.245%).
  Memory re-measured: ~855–875 B/unknown → size check now uses 1000.

2026-11-08 — Electrode toolkit (src/electrodes.py): interpolation exact for linear field;
  reference change shifts all values by one constant; passive floating electrode
  draws 0 current (1e-15 A); 11.5 um square area exact on aligned grid. PASS.

2026-11-08 — V1 disk on insulating plane, current mode, a = 6.5 um, sigma = 1.5, I = 10 uA.
  Axis 0.5a–10a vs (I/2 pi sigma a) arctan(a/z): max error 0.20%.
  Electrode potential 256.84 vs 256.41 mV (+0.17%). PASS (< 2%).

2026-11-08 — V2 spreading resistance, voltage mode, V0 = 500 mV.
  Delivered 19.467 uA vs 4 sigma a V0 = 19.500 uA (-0.17%). PASS (< 2%).
  Note: assumes all of V0 drops across the saline (no interface impedance).

2026-11-08 — V3 mode consistency (outer faces grounded).
  10 uA → 247.338 mV → 10.0000 uA; field identical. PASS (< 1%).

2026-11-08 — V4 current conservation. Flux out of boxes 2a–20a = delivered current;
  return current = −delivered. Exact to rounding. PASS (< 1%).

2026-11-08 — V7 far field. phi / (I / 2 pi sigma r) at 10a–25a: 0.995–1.000 along axis
  and substrate (full-space formula would give 0.5). Within 0.18% of the exact disk
  solution. PASS.

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

2026-11-10 — Full prediction run reproduced exactly on the MacBook (all numbers match). 22/22 tests pass.

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
2026-11-10 — Neuron models (src/neuron.py, tests/test_neuron.py), all PASS:
  rest stays at rest (drift 0.003 mV); HH single compartment matches an ODE
  solver (+40.5 mV peak at 1.535 vs 1.533 ms); squid axon conducts at
  18.7 m/s (Hodgkin & Huxley 1952: 18.8); passive cable vs exact solution
  0.065%; Mainen-Sejnowski 37 C channels (ModelDB 2488) rest -79.8 mV, peak
  +58.9 mV (ODE solver +58.9), half-width 0.57 ms, 1 um axon conducts 1.06 m/s.
  Finding: the classic HH squid model is the wrong membrane here -- its 6.3 C
  kinetics are too slow for 7-15 us current spikes, and above ~28 C it stops
  conducting (heat block). Switched to the mammalian 37 C model.
  Numerics: ion currents solved implicitly (operator splitting) -- the
  explicit version blew up at initial-segment sodium density. Field read
  along the axon with CUBIC interpolation -- linear interpolation's kinks
  made fake activating-function spikes. dt 0.5 us / 1 um compartments change
  thresholds < 0.5%.

2026-11-10 — Aim 3 full run (runs/aim3_activation.py, 30 min, PROVISIONAL
  settings). 1 um axon, Na 8000 pS/um^2, 37 C. Threshold (mV), angle 0:
                    offset 0    5     10    15    20    30 um
    |Z| 300 k, 1 um above   126   132   325   642  1052  2188
    |Z| 300 k, 5 um above   236   268   423   713  1115  2247
    |Z| 300 k, 15 um above  768   816   968  1232  1611  2730
    |Z| 600 k, 1 um above   198   209   524  1056  1765  >3000
  At 500 mV only axons within ~10 um (300 kOhm) or ~5 um (600 kOhm) of the
  pad centre fire; nothing 15 um above the chip fires. Orientation changes
  thresholds by < 20%. Beyond 10 um threshold grows ~ distance^1.7-1.8.
  Membrane: Na 2000 -> thresholds +30%; Na 30000 (initial segment) -> -25%;
  HH squid -> >3000 mV over the pad (heat-block / too-slow kinetics).
  Point source vs full model: directly over the pad the point source says
  34 mV vs 126 mV (3.7x too easy -- the real pad spreads the current); from
  5 um outward the two agree within ~20%.

2026-11-10 — Aim 2 recovery tools (src/recovery.py, tests/test_recovery.py),
  all PASS: binning, clipping + recovery flag, spike detection at the right
  sample, physics fit with the true pattern leaves only noise (4.90 uV rms vs
  5), onset gains exact (2e-16).

2026-11-10 — Aim 2 full run (runs/aim2_recovery.py, 2.5 min). Synthetic:
  60 trials, 500 mV pulse, 629 electrodes, evoked spikes 0.15-2.5 ms, noise
  5 uV, template built from the WRONG return path (top plate) and 10% wrong
  conductivity. Amplifier range +-3 mV (not documented; also 1 and 10 mV).
    Blanking (1 ms): finds 0% of evoked spikes before 0.5 ms, 17% at 0.5-1 ms.
    Average-over-trials template: 26 false detections /electrode/s -- it
      averages the time-locked evoked spikes into the template.
    Physics template (scale + uniform offset + time constant, 3 numbers):
      scenario A (well-wall truth): same as the perfect template
      (99-100% found, 2.6 false/el/s vs 2.6).
      scenario B (periphery ring + 20 kOhm stimulator): finds them, but
      65 false/el/s -- the ring's spatial pattern is not a smooth correction.
      Adding x, y, x^2-y^2, xy shapes did NOT help (70).
    Physics TIME course + per-electrode size measured in the first 100 us
      (before any neuron can respond), averaged over trials: scenario B
      2.1 false/el/s vs perfect 1.6, same spikes found. Works in every case.
  Saturation: at 1-5 pitches in scenario B with a +-1 mV range, every method
  (even the perfect template) finds only 5% of spikes before 0.5 ms -- the
  amplifier is clipped. Nothing can fix that; the input range decides it.
  Only the CHARGE (scale x capacitance) is identifiable at 20 kHz, not C alone.

---- Nov 11, 2026 (research plan v13 computation) ----

2026-11-11 — Anisotropic solver (Part 1 step 2; tests/test_anisotropic.py),
  PASS. Exact anisotropic point source, octant, three grids (ratio 2):
  max error 0.87% -> 0.23% -> 0.06% at 1:1:1, 3:1:1 and 9:1:1 alike;
  observed order 1.88 then 1.96 (second order). Three equal components give
  the isotropic answer exactly (difference 0).
  MaxOne at 9:1:1 in the array plane (geometric mean held at 1.5 S/m;
  runs/anisotropic_maxone.py): artifact along the fast axis x1.79 at
  1 pitch -> x2.13 at 10 -> x2.85 at 100 (return path); across it x0.70.
  Far-field check: exact half-space value 9^(1/3) = 2.08 along, 0.69 across.
  VERIFICATION only -- no anisotropic medium is measured.

2026-11-11 — Saturation map, whole array (step 18; runs/saturation_map.py).
  500 mV, ideal source (upper bound), |Z| 300 kOhm, 0.5 ms recovery:
    +-1 mV: 100% of electrodes saturate (up to 2.2 mm away)
    +-3 mV: 40% (top plate) - 65% (well wall), up to 1.0-1.4 mm
    +-10 mV: 5-6%, up to ~0.4 mm;  +-30 mV: 0.6%, up to ~0.13 mm
  The input range decides how much of the array Aim 2 can use at all.

2026-11-11 — Phantom predictions, DRAFT with default/ASSUMED geometry
  (runs/phantom_predict.py; NOT the committed set -- rerun with the real
  setup file and tag phantom-prediction-v1 before Nov 21).
  Half-tank symmetry checked against the full tank: 0.02% difference.
  G = phi sigma / I (1/m), full model, scope ground = return wire:
    1 mm 145.7 (grid U 1.9%, position U 10%), 5 mm 64.6, 20 mm 41.4 (U 3.2%)
  Disk on an infinite plane (no return) gives 113.5 / 31.2 / 7.95: at 20 mm,
  81% of the reading is the RETURN WIRE's own spreading resistance.
  -> the return electrode must be modelled; a separate far reference wire
     would remove most of that offset (predicted alongside).
  Position uncertainty (+-0.1 mm) dominates at 1 mm (10%).
  Part 7: readings on the container surface vary only +-2.4-2.7% between
  positions (the return dominates); a line crossing the agar boundary shows
  the interface directly (offset-free differences).
  Analysis chain tested on synthetic data (tests/test_phantom_compare.py).

2026-11-11 — Aim 2 calibration-budget study (steps 19-30; runs/aim2_budget.py,
  ~17 min). Synthetic, generated noise; split frozen in notes/aim2_split.json
  (seed 777). True world: well-wall return, 450 kOhm, 10 kOhm stimulator.
  Recall of evoked spikes < 1 ms at <= 2 false/el/s, +-3 mV, 300 Hz high-pass:
    physics + offset          0.96 at every budget (1-20 pulses); earliest 0.25 ms
    low-rank empirical        0.82-0.89 (never 0.9); earliest 1.0-2.0 ms
    per-channel average       0.43 -> 0.31 (absorbs time-locked evoked spikes)
    physics (scale, tau)      0.84;  no correction 0.85;  blanking 0.13
  Without the high-pass the artifact is only the ~10 us current spikes:
  no correction already finds 0.91; physics + offset 0.94 (earliest 0.5 ms).
  12 wrong-model templates: periphery-ring templates fail (0.84); failures
  attributed to the spatial pattern; 8/12 name the true (well-wall) return.
  Bug found and fixed before the final run: blanked samples had been treated
  like saturated ones (dropped from recall), flattering blanking.

2026-11-11 — Aim 3 maps (steps 37-42; runs/aim3_maps.py, ~42 min). Axon with
  an initial segment (Na 30000 over 5-45 um), 1 um above the chip:
  fires if it passes within ~10-12 um of the pad. Full vs point field:
  Jaccard 0.87-1.00, centroid shift <= 1.1 um (<= 0.06 pitches); small
  differences at 400 and 600 mV exceed the threshold grid uncertainty (median
  0.6%). Pad under the initial segment: 106 mV vs 125 mV under the shaft
  (agrees with Ronchi et al. 2019). Pulse width 100-400 us: < 10% effect.
  Threshold ~ distance^1.73 (10-30 um).
