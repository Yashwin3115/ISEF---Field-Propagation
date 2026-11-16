# Phantom pass criteria (fixed BEFORE any measurement)

Written Nov 15, 2026. Committed under the same git tag as the blind
predictions (`phantom-prediction-v1`). The numbers these rules use are the
constants in `src/phantom_criteria.py` (RULES). `runs/phantom_predict.py`
resolves them for the measured geometry into `pass_criteria.json` next to the
predictions (and records this file's SHA-256 there); `runs/phantom_compare.py`
only applies them. Nothing in this file may change after the tag.

## Words
- **Reading:** one row of the measurements CSV: the plateau of N averaged
  traces at one position, in one layout, under one referencing condition, in
  one session.
- **Band:** the committed 95% prediction band for that reading. It comes from
  a Monte Carlo over the measured inputs (probe position and height, electrode
  diameter, reference-wire position) with the grid uncertainty drawn in the
  same samples.
- **Covered:** the reading lies inside its band after the band is widened by
  what was measured on the day: each half-width in quadrature with
  2·sqrt(u_sigma² + u_I² + u_gain²)·|V_pred|, then by 2 × the reading's
  noise floor (from `runs/scope_plateau.py`, quantization included).
- **k_min(n):** the smallest number of covered readings out of n that is
  still consistent with a true 95% band: the largest k with
  P(X ≥ k | n, 0.95) ≥ 0.95 (binomial). Examples: 11 of 13; 23 of 26.
- **Simple model:** the model the full model is tested against.
  - Distance series: the point-source formulas. The plan's control is
    I/(4πσr). The fairest one for a reading against the reference wire is the
    "referenced" formula: the same point source, minus its own value at the
    reference wire.
  - Plate ratio: "exactly 2 at every distance".
  - Return ratios: "exactly 1 at every distance" (no dependence on where the
    return electrode is).
  - Part 7: "no interface" (the ratio a uniform medium would give at the same
    four points).
- **Discriminating position:** a position where, at commit time, the simple
  model's prediction lies outside the full model's band widened by the
  expected measurement uncertainty. These positions are listed in
  `pass_criteria.json` before any data exist.

## Rule 1: does the full model pass?
For each layout and referencing condition, pooled over sessions and pulse
settings: the full model **passes** if at least k_min(n) of its n readings are
covered. Derived ratios use the same rule on their own committed bands.

## Rule 2: does the simple model fail?
Only readings at the simple model's discriminating positions count. The
simple model **fails** if the full model is closer to the data in at least
75% of those readings and, when there are 6 or more of them, a one-sided
paired Wilcoxon signed-rank test on the absolute errors gives p < 0.05.
If a simple model has fewer than 3 discriminating positions in a layout, the
layout is declared **not able to test it**. That is decided at commit time.

## The four outcomes (all reported as they fall)
| Full model | Simple model | Stated result |
|---|---|---|
| passes | fails | The hypothesis is supported in this experiment. |
| passes | does not fail | The experiment could not tell the models apart. |
| passes | not testable | The full model is consistent with the data; no comparison is claimed. |
| does not pass | (any) | The full model is rejected here. Reported as a failed blind prediction. |

A later refit (for example an adjusted geometry) is allowed only as a
separate result labelled "fitted", shown beside the blind result, never in
its place.

## Checks with no model input
- **Reciprocity** (any layout): swapping the driven pair and the measured
  pair changes V/I by no more than sqrt(0.02² + (2·√2·noise/|V|)²).
- **A − B difference:** covered by its band (Rule 1), and its spread across
  positions within a session is no more than
  sqrt(0.02² + (2·√2·2·noise/|V_A − V_B|)²). The prediction is zero spread.
- **Two saline concentrations:** for the same layout, position and condition,
  (V·σ/I) in the two salines has ratio 1 within the committed repositioning
  band widened by both sessions' uncertainties (Rule 1 on those ratios).
- **Linearity:** V/I at two currents agrees within the two readings' bands.

## Part 7
The full model **passes** if the measured gradient ratio lies inside the
committed band (positions and grid), corrected to the measured
conductivities and widened by the session's noise and conductivity
uncertainties. "No interface" **fails** if its ratio lies outside the
measured ratio's band of the same width. The gap between the
finite-difference prediction and the ideal ratio σ_out/σ_in is reported
separately and is not a pass condition.

## Interface (separate tag, `phantom-interface-prediction-v1`)
The pulse-transient prediction is made from the impedance sweep alone and
tagged before any pulse trace is opened. It **passes** if the measured rise
time and the plateau droop each lie inside their committed bands.

## What is not a pass condition
The decay exponent, RMSE values and figures are descriptive. They are
reported, but no verdict depends on them.
