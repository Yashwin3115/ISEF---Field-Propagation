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