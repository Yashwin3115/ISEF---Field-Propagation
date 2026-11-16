"""Anisotropic phantom (experiment 8): blind prediction, three ways. Tag it
as its OWN prediction (aniso-prediction-v1) before the block is measured.

    python runs/aniso_predict.py notes/phantom_setup.json --label aniso-prediction-v1
    python runs/aniso_predict.py --quick             coarse grids, testing only
                                                     (~30 min, 8-9 GB: close other programs first)
    options: --fine7   add the h = 1/7 mm grid (27 million nodes, ~24 GB: does NOT fit on a 16 GB machine)

For each block variant in the setup file (anisotropic.variants):
  isotropic   no block at all
  tensor      the block as a uniform medium with the unit-cell conductivities
  resolved    every plate, gap and slot (h = 1/3 and 1/5 mm: the geometry is
              exact on every grid, see src/aniso_phantom.py)
at probe positions along the plates (in the central gap) and across them (at
gap centres), tip-averaged, as G = phi sigma / I and in mV per uA.
Every value has a band: grid uncertainty and the +-0.1 mm probe position,
combined at 95%. Grid uncertainty from two grids: 3 x the change between them
/ (r - 1), i.e. safety factor 3 with first-order convergence assumed (three
grids with --fine7: Celik's method).

Reported besides the predictions:
  where the tensor and resolved models SPLIT by more than their combined grid
  uncertainty (the limit of treating the block as a uniform medium)
  the anisotropy seen by the probes: reading along / reading across at the
  same distance, for all three models
  node counts and the memory each grid needs
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phantom import load_setup
from aniso_phantom import solve_block, effective_tensor, celik, aniso_geometry

args = [a for a in sys.argv[1:] if not a.startswith("--")]
QUICK = "--quick" in sys.argv
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else \
        ("aniso_quick" if QUICK else "aniso-prediction-v1")
if "--label" in sys.argv:
    args = [a for a in args if a != label]
setup_path = args[0] if args else None
setup = load_setup(setup_path)
an = setup["anisotropic"]
NS = [1, 3] if QUICK else ([3, 5, 7] if "--fine7" in sys.argv else [3, 5])        # coarse -> fine
OUT = ROOT / "results" / "aniso_predictions" / label
FIG = ROOT / "figures" / "aniso_predictions" / label
if OUT.exists() and any(OUT.iterdir()) and not QUICK:
    sys.exit(f"{OUT.relative_to(ROOT)} already exists -- predictions are never overwritten. Use --label.")
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
sig = setup["part6"]["sigma_nominal_S_per_m"]
pos_sd = setup["uncertainties"]["probe_position_sd_mm"]
t0 = time.time()
print(f"setup: {setup_path or 'built-in defaults (ASSUMED geometry)'}; label {label}; block grids h = "
      + ", ".join(f"1/{n} mm" for n in NS) + ("  [QUICK]" if QUICK else ""))
BYTES = 900                                              # measured on the MacBook, per unknown

def band(vals, dpos):
    """vals: list over grids (coarse -> fine). Returns fine value, relative grid
    uncertainty, and the 95% half-width (grid GCI with 2 x position SD)."""
    f = vals[-1]
    if len(vals) >= 3:
        U = celik(vals[-1], vals[-2], vals[-3], 1 / NS[-1], 1 / NS[-2], 1 / NS[-3])
    else:                    # two grids: Roache's GCI with safety factor 3 and FIRST order assumed
        r = NS[-1] / NS[-2]  # (the order this solver shows at electrode edges), so U = 3 |change| / (r - 1)
        U = 3.0 * np.abs(vals[-1] - vals[-2]) / np.abs(f) / (r - 1)
    return f, U, np.sqrt((U * np.abs(f)) ** 2 + (2 * dpos) ** 2)

result = {"label": label, "quick": QUICK, "created": time.strftime("%Y-%m-%d %H:%M"), "setup": setup,
          "setup_file": setup_path, "grid_n": NS, "sigma_nominal_S_per_m": sig, "variants": {}}
first_variant = next(iter(an["variants"]))
iso = [solve_block(setup, first_variant, n, "isotropic", pos_sd_mm=pos_sd) for n in NS]
dx, dy = np.array(an["along_distances_mm"]), np.array(an["across_distances_mm"])
fig, axs = plt.subplots(len(an["variants"]), 2, figsize=(12, 4.4 * len(an["variants"])), layout="constrained",
                        squeeze=False)
for vi, variant in enumerate(an["variants"]):
    v, _, _ = aniso_geometry(setup, variant)
    T = effective_tensor(v["slot_pitch_mm"], slot_mm=v["slot_width_mm"], staggered=v.get("staggered", True),
                         n=5 if QUICK else 15)
    print(f"\n{variant}: {v['slot_width_mm']:g} mm slots every {v['slot_pitch_mm']:g} mm, "
          f"{'staggered' if v.get('staggered', True) else 'aligned'}"
          f"; effective conductivities (x, y, z) / sigma = {T[0]:.3f}, {T[1]:.4f}, {T[2]:.3f} -> "
          f"{T[0]/T[1]:.2f} : 1 along : across")
    runs = {"isotropic": iso,
            "tensor": [solve_block(setup, variant, n, "tensor", T, pos_sd_mm=pos_sd) for n in NS],
            "resolved": [solve_block(setup, variant, n, "resolved", pos_sd_mm=pos_sd) for n in NS]}
    rec = {"slot_pitch_mm": v["slot_pitch_mm"], "slot_width_mm": v["slot_width_mm"], "effective_tensor": list(T), "ratio_along_over_across": T[0] / T[1],
           "ideal_reading_ratio_sqrt": float(np.sqrt(T[0] / T[1])),
           "nodes_by_grid": [r["nodes"] for r in runs["resolved"]],
           "memory_GB_by_grid": [r["unknowns"] * BYTES / 1e9 for r in runs["resolved"]],
           "along": {"distance_mm": dx.tolist()}, "across": {"distance_mm": dy.tolist()}}
    B = {}
    for line, key, dk, dist in (("along", "G_along", "dG_pos_along", dx), ("across", "G_across", "dG_pos_across", dy)):
        for m, rr in runs.items():
            f, U, hw = band([r[key] for r in rr], rr[-1][dk])
            B[(line, m)] = (f, U, hw)
            rec[line][m] = {"G_per_m": f.tolist(), "U_num_rel": U.tolist(), "band_half_width_per_m": hw.tolist(),
                            "mV_per_uA": (f / sig / 1e3).tolist(), "G_by_grid": [r[key].tolist() for r in rr]}
        fr, Ur, _ = B[(line, "resolved")]; ft, Ut, _ = B[(line, "tensor")]
        diff = np.abs(fr - ft)
        unc = np.sqrt((Ur * fr) ** 2 + (Ut * ft) ** 2)
        split = diff > unc
        rec[line]["tensor_vs_resolved"] = {"rel_difference": (ft / fr - 1).tolist(), "split": split.tolist(),
                                           "combined_grid_uncertainty_rel": (unc / fr).tolist()}
        agree = [float(d) for d, s_ in zip(dist, split) if not s_]
        rec[line]["agree_from_mm"] = (float(dist[np.flatnonzero(split)[-1] + 1])
                                      if split.any() and not split[-1] else (None if split[-1] else float(dist[0])))
        print(f"  {line} the plates   " + " ".join(f"{d:6g}" for d in dist) + "  mm")
        for m in ("isotropic", "tensor", "resolved"):
            print(f"    {m:10s} mV/uA  " + " ".join(f"{q/sig/1e3:6.4f}"[:6] for q in B[(line, m)][0]))
        print(f"    resolved band +-% " + " ".join(f"{q/abs(fv)*100:6.1f}" for q, fv in zip(B[(line, 'resolved')][2], fr)))
        print(f"    tensor/resolved-1 " + " ".join(f"{q*100:+6.1f}" for q in ft / fr - 1) + "  %")
        print(f"    split > grid unc. " + " ".join(f"{'yes' if s_ else 'no':>6}" for s_ in split))
        a = rec[line]["agree_from_mm"]
        print(f"    -> tensor and resolved models " + (f"agree from {a:g} mm outward" if a is not None else
              "differ by more than the grid uncertainty at EVERY probed distance"))
    common = [d for d in dx if d in dy]
    rec["reading_ratio_along_over_across"] = {"distance_mm": common}
    for m in ("isotropic", "tensor", "resolved"):
        ra = [float(B[("along", m)][0][list(dx).index(d)] / B[("across", m)][0][list(dy).index(d)]) for d in common]
        rec["reading_ratio_along_over_across"][m] = ra
    print("  reading along / reading across at the same distance (condition A):")
    for m in ("isotropic", "tensor", "resolved"):
        print(f"    {m:10s} " + "  ".join(f"{d:g}mm {q:.2f}" for d, q in zip(common, rec['reading_ratio_along_over_across'][m])))
    print(f"  grids (resolved): " + "; ".join(f"h 1/{n} mm: {r['nodes']/1e6:.2f}M nodes, ~{r['unknowns']*BYTES/1e9:.1f} GB"
                                            for n, r in zip(NS, runs["resolved"])))
    result["variants"][variant] = rec
    for ax, line, dist in ((axs[vi, 0], "along", dx), (axs[vi, 1], "across", dy)):
        for m, c in (("isotropic", "0.5"), ("tensor", "#eb6834"), ("resolved", "#2a78d6")):
            f, U, hw = B[(line, m)]
            ax.fill_between(dist, (f - hw) / sig / 1e3, (f + hw) / sig / 1e3, color=c, alpha=0.25, lw=0)
            ax.semilogy(dist, f / sig / 1e3, "o-", color=c, lw=1.6, ms=3.5, label=m)
        ax.axvline(19.5, color="0.6", ls="--", lw=1)
        ax.set_xlabel(f"distance {line} the plates (mm); dashed: edge of the block")
        ax.set_ylabel("mV per uA (condition A)")
        ax.set_title(f"{variant} ({v['slot_width_mm']:g} mm slots every {v['slot_pitch_mm']:g} mm, "
                     f"{T[0]/T[1]:.1f}:1): {line} the plates", fontsize=10)
        ax.grid(True, which="both", color="0.92", lw=0.5); ax.legend(frameon=False, fontsize=8)
    del runs
fig.suptitle(f"Anisotropic block: isotropic vs effective tensor vs fully resolved [{label}]"
             + (" -- QUICK" if QUICK else "") + ("" if setup_path else " -- ASSUMED GEOMETRY"))
fig.savefig(FIG / "aniso_predictions.png", dpi=200)
(OUT / "predictions.json").write_text(json.dumps(result, indent=2))
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
if not QUICK and setup_path:
    print(f"NEXT, BEFORE MEASURING THE BLOCK: git add {OUT.relative_to(ROOT)} {FIG.relative_to(ROOT)} && "
          f"git commit -m 'Anisotropic phantom blind prediction' && git tag {label} && git push && git push --tags")
