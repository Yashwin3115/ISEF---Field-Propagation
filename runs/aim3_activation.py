"""Aim 3: which axons does a stimulation pulse activate?

Field: the provisional MaxOne model (top-plate return), potential per ampere
everywhere in the fluid, from the solver. Time course: the interface circuit
for a biphasic pulse (positive phase first, 200 us per phase), at both ends of
MaxWell's impedance range for the chip.

Axons: straight, unmyelinated, 1 um diameter, 2 mm long, lying parallel to
the chip at height z0 (default 1 um). Membranes (src/neuron.py, verified in
tests/test_neuron.py):
  main         Mainen-Sejnowski 37 C channels (ModelDB 2488), Na 8000 pS/um^2
  sensitivity  Na 2000 (ordinary axon) and 30000 pS/um^2 (initial-segment
               density, the most excitable case)
  reference    Hodgkin-Huxley squid axon at 6.3 C (the classic model; too
               slow and too cold for mammalian neurons -- included to show
               how much the membrane model matters)
An axon "fires" if an action potential reaches 500 um along it in either
direction.

For each axon position (sideways offset from the pad centre, y0) and
orientation (0 deg = along the pad's 8.75 um side, 90 deg = along the 12 um
side, 45 deg), find the lowest pulse amplitude that makes it fire. Repeat
with the ideal point-source field I/(2 pi sigma r) to see how the simpler
model would change the conclusion.

Field values along each axon come from CUBIC interpolation of the solver's
field: linear interpolation puts kinks at grid lines, which the activating
function (a second derivative) turns into large spurious spikes.

    python runs/aim3_activation.py            full run (~30-60 min on the MacBook)
    python runs/aim3_activation.py --quick    coarse, for testing (~2 min)

Outputs: results/runs/aim3[_quick]/summary.json, figures/runs/aim3[_quick]/*.png
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import transfer_run, electrode_xy, CHIPS, PROVISIONAL, um, mm
from interface import capacitance_from_impedance, biphasic, circuit_current
from neuron import Axon, HHMembrane, MainenMembrane, straight_axon_points, thresholds

QUICK = "--quick" in sys.argv
NAME = "aim3_quick" if QUICK else "aim3"
OUT = ROOT / "results" / "runs" / NAME
FIG = ROOT / "figures" / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

GRID = (dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK
        else dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm))
D_AXON = 1e-4                      # cm (1 um)
LENGTH = 2 * mm
N_COMP = 1000                      # 2 um compartments
DT = 0.001                         # ms
T_END = 3.0                        # ms
T_PULSE = 0.1                      # ms, pulse onset
PHASE = 200e-6                     # s
Z0_LIST = [1 * um] if QUICK else [1 * um, 5 * um, 15 * um]
Y0_LIST = (np.array([0, 10, 25, 50, 100, 200]) if QUICK else
           np.array([0, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300])) * um
ANGLES = [0, 45, 90]
AMP_MAX = 3.0                      # V; above this, threshold reported as "> 3 V"

t0 = time.perf_counter()
p = dict(PROVISIONAL)
chip = CHIPS[p["chip"]]
sx, sy = electrode_xy(*p["stim_col_row"])
r = transfer_run({}, verbose=True, **GRID)
x, y, z, phi = r["x_grid"], r["y_grid"], r["z_grid"], r["phi_per_A"]
R_sol, sigma = r["R_solution_ohm"], p["sigma_S_per_m"]
print(f"field solved in {time.perf_counter()-t0:.0f} s; pad-to-return {R_sol/1e3:.2f} kOhm")

# smooth (cubic) interpolation of the field on a box around the pad
half = LENGTH / 2 + 0.3 * mm
bx = (x > sx - half) & (x < sx + half)
by = (y > sy - half) & (y < sy + half)
bz = z < 80 * um
field = RegularGridInterpolator((x[bx], y[by], z[bz]), phi[np.ix_(bx, by, bz)],
                                method="cubic", bounds_error=True)
del phi, r

# current per time step for a 1 V pulse, averaged over each step
def unit_current(C):
    steps = int(round(T_END / DT))
    sub = 20
    t = (np.arange(steps * sub) + 0.5) * (DT / sub) * 1e-3            # s
    i = circuit_current(t, biphasic(1.0, PHASE, T_PULSE * 1e-3), R_sol, C)
    return i.reshape(steps, sub).mean(axis=1)

C_cases = {f"|Z| {z/1e3:.0f} kOhm": capacitance_from_impedance(z) for z in chip["z_1kHz"]}

def make_axon(mem_key):
    if mem_key == "HH squid 6.3C":
        return Axon(n=N_COMP, dx_cm=LENGTH / N_COMP * 100, d_cm=D_AXON, Ra_ohm_cm=35.4,
                    dt_ms=DT, membrane=HHMembrane(6.3))
    g = float(mem_key.split()[1])
    return Axon(n=N_COMP, dx_cm=LENGTH / N_COMP * 100, d_cm=D_AXON, Ra_ohm_cm=150.0,
                dt_ms=DT, membrane=MainenMembrane(gna_pS_um2=g))

MAIN = "Na 8000 37C"
SENS = [] if QUICK else ["Na 2000 37C", "Na 30000 37C", "HH squid 6.3C"]
check = [N_COMP // 2 - 250, N_COMP // 2 + 250]          # 500 um either side

# build every axon's field profile: full model and point source
cases = []
for z0 in Z0_LIST:
    for ang in ANGLES:
        for y0 in Y0_LIST:
            # closest point to the pad centre is (sx, sy + y0); axon direction at angle ang
            # measured from x; for 90 deg, the offset is taken along x instead
            th = np.deg2rad(ang)
            u = np.array([np.cos(th), np.sin(th)])
            nrm = np.array([-np.sin(th), np.cos(th)])
            centre = np.array([sx, sy]) + y0 * nrm
            pts, s = straight_axon_points(np.array([*centre, z0]), u, LENGTH, N_COMP)
            full = field(pts)
            rr = np.sqrt((pts[:, 0] - sx) ** 2 + (pts[:, 1] - sy) ** 2 + pts[:, 2] ** 2)
            point = 1 / (2 * np.pi * sigma * rr)
            cases.append({"z0_um": z0 / um, "angle_deg": ang, "y0_um": y0 / um,
                          "full": full, "point": point})
print(f"{len(cases)} axon positions x 2 field models x {len(C_cases)} impedances")

results = []
jobs = [(MAIN, cname, model) for cname in C_cases for model in ["full", "point"]]
c_low = list(C_cases)[0]                                   # low-impedance end
jobs += [(mk, c_low, "full") for mk in SENS]
for mem_key, cname, model in jobs:
    t1 = time.perf_counter()
    axon = make_axon(mem_key)
    iu = unit_current(C_cases[cname])
    sel = cases if mem_key == MAIN else [c for c in cases if c["z0_um"] == Z0_LIST[0] / um]
    Phi = np.column_stack([c[model] for c in sel])
    th = thresholds(axon, Phi, iu, T_END, check, amp_max_V=AMP_MAX, iterations=11)
    for c, v in zip(sel, th):
        results.append({"membrane": mem_key, "impedance": cname, "model": model,
                        "z0_um": c["z0_um"], "angle_deg": c["angle_deg"], "y0_um": c["y0_um"],
                        "threshold_mV": None if np.isnan(v) else float(v * 1e3)})
    print(f"  {mem_key:14s} {cname}, {model:5s} field: {time.perf_counter()-t1:.0f} s", flush=True)

def thr(cname, model, z0, ang, mem_key=MAIN):
    rows = [q for q in results if q["membrane"] == mem_key and q["impedance"] == cname
            and q["model"] == model and q["z0_um"] == z0 / um and q["angle_deg"] == ang]
    rows.sort(key=lambda q: q["y0_um"])
    return (np.array([q["y0_um"] for q in rows]),
            np.array([np.nan if q["threshold_mV"] is None else q["threshold_mV"] for q in rows]))

# activation distance: farthest offset that still fires at 400 / 500 / 600 mV
reach = []
for mem_key, cname, model in jobs:
    for z0 in (Z0_LIST if mem_key == MAIN else Z0_LIST[:1]):
            for ang in ANGLES:
                yy, tt = thr(cname, model, z0, ang, mem_key)
                for A in [400, 500, 600]:
                    ok = np.where(tt <= A)[0]
                    reach.append({"membrane": mem_key, "impedance": cname, "model": model,
                                  "z0_um": z0 / um,
                                  "angle_deg": ang, "amplitude_mV": A,
                                  "max_offset_firing_um": float(yy[ok].max()) if len(ok) else None,
                                  "all_inner_offsets_fire": bool(len(ok) == ok.max() + 1) if len(ok) else None})

(OUT / "summary.json").write_text(json.dumps({
    "status": "PROVISIONAL settings (see maxone.PROVISIONAL)", "quick": QUICK,
    "settings": {**{k: v for k, v in p.items()}, "R_pad_to_return_ohm": R_sol,
                 "axon_diameter_um": D_AXON * 1e4, "axon_length_mm": LENGTH / mm,
                 "compartments": N_COMP, "dt_ms": DT, "main_membrane": MAIN,
                 "sensitivity_membranes": SENS,
                 "pulse": "biphasic, positive first, 200 us/phase", "grid": GRID},
    "thresholds": results, "reach": reach}, indent=2, default=float))

# ---------------- print ----------------
cname0 = list(C_cases)[0]
print(f"\nthreshold (mV) vs sideways offset, {MAIN}, axon height {Z0_LIST[0]/um:.0f} um, {cname0}")
print(f"{'offset um':>9} " + " ".join(f"{f'{a} deg full':>12} {f'point':>7}" for a in ANGLES))
for k, y0 in enumerate(Y0_LIST):
    cells = []
    for a in ANGLES:
        _, tf = thr(cname0, "full", Z0_LIST[0], a)
        _, tp = thr(cname0, "point", Z0_LIST[0], a)
        f = lambda v: f"{v:7.0f}" if np.isfinite(v) else "  >3000"
        cells.append(f"{f(tf[k]):>12} {f(tp[k]):>7}")
    print(f"{y0/um:9.0f} " + " ".join(cells))
print("\nfarthest offset that fires at 500 mV (um), full field:")
for q in reach:
    if q["model"] == "full" and q["amplitude_mV"] == 500:
        print(f"  {q['membrane']:14s} {q['impedance']:14s} height {q['z0_um']:4.0f} um  "
              f"angle {q['angle_deg']:3d}:  {q['max_offset_firing_um']}")
if SENS:
    print(f"\nmembrane sensitivity, angle 0, height {Z0_LIST[0]/um:.0f} um, {c_low}: threshold (mV)")
    print(f"{'offset um':>9} " + " ".join(f"{m:>15}" for m in [MAIN] + SENS))
    for k, y0 in enumerate(Y0_LIST):
        cells = []
        for m in [MAIN] + SENS:
            _, tt = thr(c_low, "full", Z0_LIST[0], 0, m)
            cells.append(f"{tt[k]:15.0f}" if np.isfinite(tt[k]) else f"{'>3000':>15}")
        print(f"{y0/um:9.0f} " + " ".join(cells))

# ---------------- figures ----------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "0.4", "axes.labelcolor": "0.15",
                     "xtick.color": "0.35", "ytick.color": "0.35"})
colors = {0: "#2a78d6", 45: "#eb6834", 90: "#1baf7a"}
tag = "PROVISIONAL" + (" (quick)" if QUICK else "")
fig, axs = plt.subplots(1, len(C_cases), figsize=(11, 4.3), sharey=True)
for ax, cname in zip(np.atleast_1d(axs), C_cases):
    for a in ANGLES:
        yy, tf = thr(cname, "full", Z0_LIST[0], a)
        _, tp = thr(cname, "point", Z0_LIST[0], a)
        ax.plot(np.maximum(yy, 1), tf, "-o", ms=3.5, lw=1.8, color=colors[a], label=f"{a}°, full model")
        ax.plot(np.maximum(yy, 1), tp, "--", lw=1.2, color=colors[a], alpha=0.8, label=f"{a}°, point source")
    for A in [400, 600]:
        ax.axhline(A, color="0.75", lw=0.8)
    ax.axhspan(400, 600, color="0.95", lw=0)
    ax.text(1.1, 610, "Dr. Muller's 400-600 mV", fontsize=7.5, color="0.4")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sideways offset of the axon from the pad centre (µm; 0 plotted at 1)")
    ax.set_title(f"{cname} interface", fontsize=9.5)
    ax.grid(True, which="both", color="0.92", lw=0.5)
np.atleast_1d(axs)[0].set_ylabel("threshold pulse amplitude (mV)")
np.atleast_1d(axs)[0].legend(frameon=False, fontsize=7.5, ncol=2)
fig.suptitle(f"Aim 3: which axons fire? {MAIN} membrane, 1 µm axon, {Z0_LIST[0]/um:.0f} µm above the chip; {tag}",
             fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "thresholds.png", dpi=200); plt.close(fig)
print(f"\ntotal {time.perf_counter()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
