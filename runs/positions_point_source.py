"""Three stimulation positions x three return paths, and how wrong the ideal
point-source formula is at every electrode.

Positions (col, row):
    centre      (110, 60)   next to the middle of the array
    off-centre  (55, 30)    a quarter of the way in from a corner
    near edge   (2, 60)     two electrodes from the left edge

Model versions compared at every one of the 26,400 electrodes:
    1. ideal point source on an insulating plane, I / (2 pi sigma r)
    2. full model: real pad, 19 mm well, fluid depth, return path   (this solver)
(Version 3, the interface, scales version 2 in time and does not change the
spatial pattern; see interface.py.)

Uses the finest production grid (n_pad 8, growth 1.10). Takes ~10-15 minutes
and up to ~6.5 GB on the MacBook. Provisional settings, like v1.

    python runs/positions_point_source.py            full run
    python runs/positions_point_source.py --quick    coarse grid, for testing

Outputs: results/runs/positions/ (summary.json, transfer_<position>_<return>.npz)
         figures/runs/positions/ (falloff_positions.png, point_source_error.png)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import transfer_run, RETURNS, PITCH, N_COLS, um, mm

QUICK = "--quick" in sys.argv
name = "positions_quick" if QUICK else "positions"
OUT = ROOT / "results" / "runs" / name
FIG = ROOT / "figures" / "runs" / name
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GRID = (dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK
        else dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm))
POSITIONS = {"centre": (110, 60), "off-centre": (55, 30), "near edge": (2, 60)}
COLORS = {"top_plate": "#2a78d6", "well_wall": "#eb6834", "periphery": "#1baf7a"}
POS_COLORS = {"centre": "#2a78d6", "off-centre": "#eb6834", "near edge": "#1baf7a"}
BINS_PITCHES = [(1, 2), (2, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 250)]

t_all = time.perf_counter()
results = {}
for pos, cr in POSITIONS.items():
    for ret in RETURNS:
        t0 = time.perf_counter()
        r = transfer_run({"return": ret, "stim_col_row": cr}, verbose=False, **GRID)
        for k in ["x_grid", "y_grid", "z_grid", "phi_per_A"]:
            r.pop(k)
        sigma = r["params"]["sigma_S_per_m"]
        d = r["distance_m"]
        with np.errstate(divide="ignore"):
            point = 1 / (2 * np.pi * sigma * d)
        r["point_error"] = np.where(d > 0, point / r["transfer_ohm"] - 1, np.nan)
        results[(pos, ret)] = r
        np.savez_compressed(OUT / f"transfer_{pos.replace(' ', '_')}_{ret}.npz",
                            col=r["col"], row=r["row"], distance_m=d,
                            transfer_ohm=r["transfer_ohm"], point_source_ohm=point)
        print(f"{pos:10s} {str(cr):10s} {ret:10s} {r['grid']['unknowns']:>10,} unknowns, "
              f"{r['solver']['iterations']:3d} CG its, {time.perf_counter()-t0:5.0f} s, "
              f"R(pad->return) {r['R_solution_ohm']/1e3:.3f} kOhm", flush=True)

# ---------------- point-source error by distance ----------------
summary = {"settings": results[("centre", "top_plate")]["params"], "grid": GRID,
           "quick": QUICK, "positions": {k: list(v) for k, v in POSITIONS.items()},
           "point_source_error_percent": {}, "R_pad_to_return_ohm": {}}
print("\nideal point-source formula vs full model: median error (min to max) by distance")
print(f"{'position':10s} {'return':10s} " + " ".join(f"{f'{a}-{b} p':>17}" for a, b in BINS_PITCHES))
for (pos, ret), r in results.items():
    pk = r["distance_m"] / PITCH
    cells, store = [], []
    for a, b in BINS_PITCHES:
        sel = (pk >= a - 1e-9) & (pk < b - 1e-9)
        e = 100 * r["point_error"][sel]
        if sel.any():
            store.append({"pitches": [a, b], "n": int(sel.sum()), "median": float(np.median(e)),
                          "min": float(e.min()), "max": float(e.max())})
            cells.append(f"{np.median(e):+6.1f} ({e.min():+.0f},{e.max():+.0f})")
        else:
            store.append({"pitches": [a, b], "n": 0})
            cells.append(f"{'-':>17}")
    summary["point_source_error_percent"][f"{pos} | {ret}"] = store
    summary["R_pad_to_return_ohm"][f"{pos} | {ret}"] = r["R_solution_ohm"]
    print(f"{pos:10s} {ret:10s} " + " ".join(f"{c:>17}" for c in cells))

# ---------------- edge effect: same distance, toward vs away from the edge ----------------
print("\nnear-edge pad (2, 60): potential toward the edge vs toward the centre, same distance")
edge = {}
for ret in RETURNS:
    r = results[("near edge", ret)]
    c0, r0 = POSITIONS["near edge"]
    rows_ = []
    for k in [1, 2]:
        toward = r["transfer_ohm"][(r["col"] == c0 - k) & (r["row"] == r0)][0]
        away = r["transfer_ohm"][(r["col"] == c0 + k) & (r["row"] == r0)][0]
        rows_.append({"pitches": k, "toward_edge_ohm": float(toward), "toward_centre_ohm": float(away),
                      "difference_percent": float(100 * (toward / away - 1))})
        print(f"  {ret:10s} {k} pitch: toward edge {toward:8.1f}, toward centre {away:8.1f} Ohm "
              f"({100*(toward/away-1):+.2f}%)")
    edge[ret] = rows_
summary["edge_asymmetry"] = edge
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------- figures ----------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "0.4", "axes.labelcolor": "0.15",
                     "xtick.color": "0.35", "ytick.color": "0.35"})
tag = "PROVISIONAL" + (" (quick grid)" if QUICK else "")

# fall-off along the stimulating row, all three positions, each return
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, ret in zip(axs, RETURNS):
    for pos, cr in POSITIONS.items():
        r = results[(pos, ret)]
        sel = (r["row"] == cr[1]) & (r["col"] > cr[0])
        ax.loglog(r["distance_m"][sel] / um, r["transfer_ohm"][sel] * 1e-3, lw=1.8,
                  color=POS_COLORS[pos], label=f"{pos} {cr}")
    dd = np.geomspace(17.5, 4000, 200)
    ax.loglog(dd, 1e-3 / (2 * np.pi * summary["settings"]["sigma_S_per_m"] * dd * um), ls="--",
              color="0.45", lw=1.1, label="ideal point source")
    ax.set_title(f"return: {ret.replace('_', ' ')}", fontsize=10)
    ax.set_xlabel("distance along the row, toward +x (µm)")
    ax.grid(True, which="both", color="0.92", lw=0.5)
axs[0].set_ylabel("mV per µA")
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"Three stimulation positions; {tag}", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "falloff_positions.png", dpi=200); plt.close(fig)

# point-source error vs distance, centre position, three returns
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for ret in RETURNS:
    r = results[("centre", ret)]
    pk = r["distance_m"] / PITCH
    edges = np.geomspace(1, 250, 40)
    mids, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (pk >= a) & (pk < b)
        if sel.sum() >= 3:
            e = 100 * r["point_error"][sel]
            mids.append(np.sqrt(a * b) * PITCH / um); med.append(np.median(e))
            lo.append(np.percentile(e, 5)); hi.append(np.percentile(e, 95))
    ax.fill_between(mids, lo, hi, color=COLORS[ret], alpha=0.15, lw=0)
    ax.semilogx(mids, med, color=COLORS[ret], lw=2, label=f"return: {ret.replace('_', ' ')}")
ax.axhline(0, color="0.6", lw=0.8)
ax.set_xlabel("distance from stimulating pad (µm)")
ax.set_ylabel("point-source formula error (%)")
ax.set_title(f"How wrong is I/(2πσr)? centre pad, median and 5-95% range; {tag}", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "point_source_error.png", dpi=200); plt.close(fig)

print(f"\ntotal time {time.perf_counter() - t_all:.0f} s; wrote {OUT.relative_to(ROOT)}/ and "
      f"{FIG.relative_to(ROOT)}/")
