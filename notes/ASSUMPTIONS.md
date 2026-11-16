# Assumed values and their stated basis (research plan v16)

Every number below is an ASSUMPTION, not a measurement or a confirmed lab
setting. Each line says what it is based on and which result is sensitive to
it. Replace with confirmed values when Dr. Muller or the phantom sessions
provide them.

| Quantity | Value used | Stated basis | What depends on it |
|---|---|---|---|
| Amplifier high-pass corner | off, or 300 Hz (first order) | Ronchi et al. 2019 report noise in two bands split at 300 Hz (1-300 Hz and 300 Hz-10 kHz), which suggests 300 Hz is a band boundary on this readout. This is a stated basis, **not** confirmation that 300 Hz is what Dr. Muller's lab runs. | Aim 2: with no filter the artifact is only the ~10 us current spikes; with 300 Hz it has millisecond tails. Both are run. |
| Amplifier input range | +-1, 3, 10, 30 mV (Aim 2 main case +-3 mV) | Not documented; swept | Saturation map; which electrodes Aim 2 can use at all |
| Amplifier recovery after clipping | 0.1 / 0.5 / 2 ms (saturation map); 0.1 ms or 3 filter time constants (Aim 2) | Not documented; swept | Saturation map |
| Stimulator output resistance | 0-100 kOhm; 10 kOhm in the Aim 2 synthetic world | No published value found; plan v16 labels the range an assumption | Peak artifact: 224 -> 47 mV at the neighbour instantaneous, 42 -> 29 mV as sampled (runs/stimulator_sweep.py). Saturation at +-3 mV: 40-65% (ideal) vs 2.5-3% (100 kOhm) |
| Electrode size | 8.75 x 12 um (MaxOne User Guide); 5 x 9 um checked (Ronchi et al. 2019) | Sources disagree; probably chip variants | Transfer resistance: <1%. Peak artifact -34% with the small pad. Aim 3 thresholds up to ~2x (runs/pad_size_sensitivity.py) |
| Electrode impedance | 300-600 kOhm at 1 kHz (450 kOhm in the Aim 2 world) | MaxWell's published table for platinum black | Interface time constant and charge |
| Pulse-to-pulse variability | none / low (1%, 2%, 5 uV) / medium (3%, 5%, 10 uV) / high (10%, 10%, 25 uV): amplitude, tau, slow drift | No published pulse-to-pulse figures found; swept from zero | Aim 2 fixed vs per-pulse arms |
| Spike waveforms | 5 parametric classes, 50-200 uV, ~1 ms | Constructed to span published ranges; NOT digitized published templates | Aim 2 recall by waveform |
| Recording noise (synthetic) | 5 uV white, generated | Replace with phantom noise recordings (step 58) | Aim 2 detection thresholds |
| Response latency windows | direct 1.5-20 ms, synaptic from ~5 ms | Wagenaar et al. 2002, a conference abstract; no sub-millisecond window is reported (no source) | Aim 2 latency classes (step 33) |
| Phantom geometry | see notes/phantom_setup_template.json (fields marked ASSUMED) | Placeholders until the apparatus is measured (week of Nov 16) | All phantom predictions -- must be replaced before tagging |
| Phantom voltage noise after averaging | 5 uV per reading | Placeholder; replace with the session noise floor | Part 7 gradient-ratio test: 5 uV gives ~32% ratio uncertainty; ~1 uV is needed for ~10% |
| Scope and generator (phantom) | 8 bits over 8 divisions, smallest 2 mV/div (ADC step 62.5 uV), 2 scope channels, single-sample noise 150 uV; 1 generator channel, 10 V maximum, 100 kOhm series resistor | NOT CONFIRMED (Nov 15): typical values for a school 8-bit scope. Set in the `instrument` block of notes/phantom_setup.json; measure the noise with runs/scope_noise.py | Noise floor inside every band; how many traces must be averaged; which Part 7 and Aim 2 tests are possible |
| Input uncertainties for the prediction bands | probe position 0.1 mm, tip height 0.25 mm, electrode diameter 2%, reference wire 1 mm, conductivity meter 1%, temperature 0.5 C (2% per C), current 1%, scope gain 1.5% | Reading resolution of a ruler/caliper, a pocket conductivity meter and a 1% resistor; the scope's usual DC gain accuracy. Replace with the instruments' stated values | Width of every prediction band (Monte Carlo, 4000 samples) |
| Phantom layouts E2, E3 | post 1 mm x 50 mm; top sheet 100 x 100 mm, 20 mm above the electrode; ring radius 30 mm, 0.5 mm wide | Placeholders for the Nov 15 experiment list | Plate ratio and return ratios |
| Anisotropic block | 1 mm plates and gaps, 39 mm long, 10 plates each side, 40 mm high; staggered slots: 3 mm every 8 mm (2.9:1) and 1 mm every 10 mm (9.2:1) | Hub's Nov 15 proposal; slot sizes chosen by runs/aniso_design.py. Printed dimensions must be measured (a 0.1 mm error in a 1 mm gap is 10%) | Anisotropic prediction; where the tensor model stops working |
| Interface demo values | R_s 650 ohm, Q 1.5e-7, alpha 0.82, stray 80 pF | Made up, for testing the pipeline only; the real fit comes from the impedance sweep | Nothing reported |
