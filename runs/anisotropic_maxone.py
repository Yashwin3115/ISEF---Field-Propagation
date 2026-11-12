"""Research plan Part 1, step 2 (second half): the MaxOne geometry with an
anisotropic medium, 9:1:1 in the array plane, to report how much the predicted
artifact changes. This is VERIFICATION use of the anisotropic solver (the
equations solved correctly in a new setting), not a claim about any real
medium: aCSF is isotropic. The ratio is the tissue-like one quoted for white
matter.

Convention: sigma_x : sigma_y : sigma_z = 9 : 1 : 1 with the geometric mean
held at the provisional 1.5 S/m (so sigma_y = sigma_z = 0.72 S/m,
sigma_x = 6.49 S/m), and the same with the fast axis along y. Each is compared
with the isotropic 1.5 S/m solution on the same grid, at electrodes along the
x and y lines through the stimulating pad.

Usage:
    python runs/anisotropic_maxone.py          (~5-10 min, ~4 GB)
    python runs/anisotropic_maxone.py --quick  (coarse grid, ~1 min)
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
from maxone import transfer_run, PITCH, PROVISIONAL, um, mm

QUICK = "--quick" in sys.argv
label = "anisotropic_quick" if QUICK else "anisotropic"
OUT = ROOT / "results" / "runs" / label
FIG = ROOT / "figures" / "runs" / label
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
GRID = dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK else \
       dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm)

g = 9 ** (1 / 3)
CASES = {"isotropic": None,
         "9:1:1, fast axis x": (9 / g, 1 / g, 1 / g),
         "1:9:1, fast axis y": (1 / g, 9 / g, 1 / g)}

t0 = time.time()
runs = {}
for name, a in CASES.items():
    r = transfer_run({"anisotropy": a}, verbose=False, **GRID)
    runs[name] = r
    print(f"{name:20s} solved ({time.time()-t0:.0f} s); pad-to-return resistance "
          f"{r['R_solution_ohm']/1e3:.1f} kOhm")

ref = runs["isotropic"]
c0, r0 = PROVISIONAL["stim_col_row"]
lines = {"x line (same row)": (ref["row"] == r0) & (ref["col"] > c0),
         "y line (same column)": (ref["col"] == c0) & (ref["row"] > r0)}
n_steps = [1, 2, 5, 10, 20, 50, 100]
summary = {"grid": GRID, "quick": QUICK,
           "convention": "geometric mean of (sx, sy, sz) held at 1.5 S/m",
           "R_solution_kOhm": {k: v["R_solution_ohm"] / 1e3 for k, v in runs.items()},
           "ratio_to_isotropic": {}}
print(f"\nartifact relative to isotropic (same grid), at n pitches from the pad")
print(f"{'case':20s} {'line':22s} " + " ".join(f"{n:>6d}" for n in n_steps))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (lname, m) in zip(axs, lines.items()):
    idx = np.flatnonzero(m)
    d = np.round(ref["distance_m"][idx] / PITCH).astype(int)
    order = np.argsort(d); idx, d = idx[order], d[order]
    for name in list(CASES)[1:]:
        ratio = runs[name]["transfer_ohm"][idx] / ref["transfer_ohm"][idx]
        cells = [ratio[d == n][0] if np.any(d == n) else np.nan for n in n_steps]
        summary["ratio_to_isotropic"][f"{name} | {lname}"] = dict(zip(map(str, n_steps), map(float, cells)))
        print(f"{name:20s} {lname:22s} " + " ".join(f"{c:6.2f}" for c in cells))
        ax.semilogx(d * PITCH / um, ratio, lw=2, label=name)
    ax.axhline(1, color="0.5", lw=1)
    ax.set_title(lname); ax.set_xlabel("distance from the pad (um)")
    ax.grid(True, which="both", color="0.92", lw=0.5)
axs[0].set_ylabel("artifact / isotropic artifact")
axs[0].legend(frameon=False)
# far-field check: a surface source in a half space with diag(sx, sy, sz) gives
# phi = I / (2 pi sqrt(sx sy sz) sqrt(x^2/sx + y^2/sy + z^2/sz)), so along the
# fast axis the ratio to isotropic tends to g = 9^(1/3) = 2.08 and across it
# to g/3 = 0.69 (when the return is far enough not to matter).
summary["far_field_expected"] = {"along fast axis": g, "across fast axis": g / 3}
print(f"\nexpected far from the pad (unbounded half space): along fast axis {g:.2f}, "
      f"across {g/3:.2f}; the return path bends the curves at large distance")
fig.suptitle("Solver verification use: MaxOne with a 9:1:1 medium (geometric mean 1.5 S/m)"
             + (" -- QUICK grid" if QUICK else ""))
fig.tight_layout()
fig.savefig(FIG / "anisotropic_ratio.png", dpi=200)
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\ntotal {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
