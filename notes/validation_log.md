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

