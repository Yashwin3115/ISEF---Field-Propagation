# Entries to add to validation_log.md (Aims 2 and 3)

Run each step on the Mac first. Copy an entry only after your own run matches;
replace any number that differs with yours. Then delete this file.
(Numbers below are from Claude's run of the same code, Nov 10, 2026.)

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
