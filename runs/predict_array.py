"""PROVISIONAL blind predictions for stimulation on the MaxOne array.

    python runs/predict_array.py --label v2      full run: 3 return paths x 3 grids
    python runs/predict_array.py --quick         one coarse grid per return (testing only)

Output goes to results/runs/<label>/ and figures/runs/<label>/. An existing
label is never overwritten (add --overwrite to allow it), so a committed blind
prediction cannot be replaced by accident. The first committed prediction,
tag provisional-prediction-v1, lives in results/predictions/ and figures/
and is never touched by this script.

Every unconfirmed setting comes from maxone.PROVISIONAL and is written into
every output file. Re-run with the lab's values when they reply, and commit
BOTH versions before looking at any measured data.

What it computes
  1. Transfer resistance Z_k (ohm) from the stimulating pad to every one of the
     26,400 electrodes: potential at electrode k per ampere delivered,
     relative to the return. Three return paths (the unconfirmed part):
     top_plate, well_wall, periphery -- see maxone.RETURNS.
  2. Grid convergence (GCI) at selected distances, from three grids whose
     every spacing shrinks by about sqrt(2) each time.
  3. The artifact for a 400 / 500 / 600 mV biphasic pulse, 200 us per phase,
     via the interface circuit (interface.py), for the two ends of MaxWell's
     impedance range for the chip.

Outputs (committed as the blind prediction)
  results/predictions/provisional_predictions.csv   one row per electrode
  results/predictions/summary.json                  settings, GCI, interface, spread
  figures/array_map.png, array_falloff.png, array_artifact.png
"""
import sys
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import (transfer_run, PROVISIONAL, RETURNS, CHIPS, PITCH, um, mm)
from interface import (capacitance_from_impedance, biphasic, circuit_current,
                       pulse_summary, sample_average)

QUICK = "--quick" in sys.argv
if QUICK:
    LABEL = "quick"
elif "--label" in sys.argv:
    LABEL = sys.argv[sys.argv.index("--label") + 1]
else:
    sys.exit("give the run a label, e.g.  python runs/predict_array.py --label v2")
OUT = ROOT / "results" / "runs" / LABEL
FIG = ROOT / "figures" / "runs" / LABEL
if OUT.exists() and not QUICK and "--overwrite" not in sys.argv:
    sys.exit(f"{OUT.relative_to(ROOT)} already exists -- choose a new --label "
             f"(or add --overwrite if you really mean it)")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# grid levels: every spacing (pad cells, growth excess, caps) scaled by f
LEVELS = [2.0] if QUICK else [2.0, 2 ** 0.5, 1.0]
def grid_settings(f):
    return dict(n_pad=int(round(8 / f)), growth=1 + 0.10 * f,
                h_array=50 * um * f, h_far=0.5 * mm * f)

AMPLITUDES = [0.4, 0.5, 0.6]           # V (Dr. Muller: 400-600 mV)
PHASE = 200e-6                          # s (MaxLab Live default)
PROBE_PITCHES = [1, 2, 5, 10, 20, 50, 100]
COLORS = {"top_plate": "#2a78d6", "well_wall": "#eb6834", "periphery": "#1baf7a"}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def gci_three(f1, f2, f3, N1, N2, N3):
    """Celik et al. (2008) procedure for three grids with any refinement ratios.
    f1 finest. Returns (p, extrapolated, GCI_fine, monotone)."""
    h1, h2, h3 = N1 ** (-1 / 3), N2 ** (-1 / 3), N3 ** (-1 / 3)
    r21, r32 = h2 / h1, h3 / h2
    e21, e32 = f2 - f1, f3 - f2
    if e21 == 0 or e32 == 0:
        return np.nan, f1, 0.0, True
    s = np.sign(e32 / e21)
    monotone = s > 0
    p = 1.0
    for _ in range(100):
        q = np.log(abs((r21 ** p - s) / (r32 ** p - s)))
        p_new = abs(np.log(abs(e32 / e21)) + q) / np.log(r21)
        if abs(p_new - p) < 1e-10:
            break
        p = p_new
    ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    gci = 1.25 * abs((f1 - f2) / f1) / (r21 ** p - 1)
    return p, ext, gci, monotone


def probe_index(run, k):
    """Electrode k pitches from the stimulating pad along +x (same row)."""
    c0, r0 = run["params"]["stim_col_row"]
    sel = np.flatnonzero((run["col"] == c0 + k) & (run["row"] == r0))
    return int(sel[0])


# --------------------------------------------------------------------------
# 1-2. transfer resistance, three returns x three grids
# --------------------------------------------------------------------------
t_all = time.perf_counter()
runs = {}
for ret in RETURNS:
    runs[ret] = []
    for f in LEVELS:
        g = grid_settings(f)
        print(f"\n=== return '{ret}', grid level f = {f:.3f} ({g}) ===", flush=True)
        t0 = time.perf_counter()
        r = transfer_run({"return": ret}, **g)
        print(f"solved in {time.perf_counter()-t0:.0f} s, {r['solver']['iterations']} CG "
              f"iterations; R(pad->return) {r['R_solution_ohm']/1e3:.3f} kOhm; current "
              f"check {r['current_check']:.6f}", flush=True)
        for key in ["x_grid", "y_grid", "z_grid", "phi_per_A"]:
            r.pop(key)                                   # keep memory down
        runs[ret].append(r)

p = runs["top_plate"][-1]["params"]
chip = CHIPS[p["chip"]]

summary = {
    "status": p["status"],
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "code_commit": git_commit(),
    "quick_mode": QUICK,
    "label": LABEL,
    "settings": {k: v for k, v in p.items() if k != "return"},
    "chip": chip,
    "returns": RETURNS,
    "grid_levels": [grid_settings(f) | {"f": f} for f in LEVELS],
    "per_return": {},
}

for ret, rr in runs.items():
    fine = rr[-1]
    entry = {"unknowns": [r["grid"]["unknowns"] for r in rr],
             "R_solution_ohm": [r["R_solution_ohm"] for r in rr], "probes": []}
    if len(rr) == 3:
        N = [r["grid"]["unknowns"] for r in rr][::-1]          # finest first
        pR, eR, gR, mR = gci_three(*[r["R_solution_ohm"] for r in rr][::-1], *N)
        entry["R_solution_extrapolated_ohm"] = eR
        entry["R_solution_gci"] = gR
        entry["R_solution_p"] = pR
    for k in PROBE_PITCHES:
        i = probe_index(fine, k)
        vals = [r["transfer_ohm"][i] for r in rr]
        row = {"pitches": k, "distance_um": float(fine["distance_m"][i] / um),
               "Z_ohm_coarse_to_fine": [float(v) for v in vals]}
        if len(rr) == 3:
            pk, ek, gk, mk = gci_three(vals[2], vals[1], vals[0], *N)
            spread_k = (max(vals) - min(vals)) / abs(vals[-1])
            # The GCI formula is only meaningful when the three grids converge
            # steadily (same-direction changes, sensible order). When the
            # differences are at noise level it can return p = 12 or a tiny
            # GCI; then the honest uncertainty is the spread across the grids.
            reliable = bool(mk and 0.5 < pk < 4)
            row.update(p=float(pk), extrapolated_ohm=float(ek), gci=float(gk), monotone=bool(mk),
                       gci_reliable=reliable, grid_spread=float(spread_k),
                       grid_uncertainty=float(gk if reliable else spread_k))
        entry["probes"].append(row)
    summary["per_return"][ret] = entry

# --------------------------------------------------------------------------
# 3. interface: actual current and artifact
# --------------------------------------------------------------------------
C_range = [capacitance_from_impedance(z) for z in chip["z_1kHz"][::-1]]   # low Z -> high C
interface = {"C_dl_F_range": C_range, "impedance_1kHz_ohm_range": list(chip["z_1kHz"]),
             "R_buf_ohm": 0.0, "R_ct_ohm": "infinite", "phase_s": PHASE, "by_return": {}}
for ret in RETURNS:
    e = summary["per_return"][ret]
    R_sol = e.get("R_solution_extrapolated_ohm", e["R_solution_ohm"][-1])
    by_amp = []
    for A in AMPLITUDES:
        for C in C_range:
            s = pulse_summary(A, PHASE, R_sol, C)
            by_amp.append({"amplitude_V": A, "C_dl_F": C, "tau_s": s["tau_s"],
                           "peak_current_A": 2 * A / R_sol,      # at the phase reversal
                           "charge_phase1_C": s["charge_phase1_C"]})
    interface["by_return"][ret] = {"R_solution_used_ohm": R_sol, "pulses": by_amp}
summary["interface"] = interface

# spread across return paths at the probe distances (finest grid)
spread = []
for j, k in enumerate(PROBE_PITCHES):
    vals = [summary["per_return"][ret]["probes"][j]["Z_ohm_coarse_to_fine"][-1] for ret in RETURNS]
    spread.append({"pitches": k, "min_ohm": min(vals), "max_ohm": max(vals),
                   "spread_percent": 100 * (max(vals) - min(vals)) / np.mean(vals)})
summary["return_path_spread"] = spread

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

# --------------------------------------------------------------------------
# per-electrode table (finest grid)
# --------------------------------------------------------------------------
fine = {ret: runs[ret][-1] for ret in RETURNS}
ref = fine["top_plate"]
A0, R_used = 0.5, interface["by_return"]["top_plate"]["R_solution_used_ohm"]
i_peak = 2 * A0 / R_used
cols = ["col", "row", "x_um", "y_um", "distance_um"] + [f"Z_{ret}_ohm" for ret in RETURNS] + \
       ["return_spread_percent", "peak_artifact_mV_top_plate_500mV_ideal_source"]
Zs = np.column_stack([fine[ret]["transfer_ohm"] for ret in RETURNS])
spread_pct = 100 * (Zs.max(1) - Zs.min(1)) / Zs.mean(1)
table = np.column_stack([ref["col"], ref["row"], ref["x"] / um, ref["y"] / um,
                         ref["distance_m"] / um, Zs, spread_pct, Zs[:, 0] * i_peak * 1e3])
header = (f"# PROVISIONAL blind prediction, {summary['created_utc']}, code {summary['code_commit']}\n"
          f"# settings: {json.dumps(summary['settings'], default=float)}\n"
          f"# Z = potential relative to the return, per ampere delivered by the stimulating pad\n"
          f"# peak artifact assumes an ideal voltage source (R_buf = 0) at the phase reversal\n"
          + ",".join(cols))
np.savetxt(OUT / "provisional_predictions.csv", table, delimiter=",", header=header, comments="",
           fmt=["%d", "%d", "%.3f", "%.3f", "%.3f"] + ["%.6g"] * (len(RETURNS) + 2))

# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "0.4", "axes.labelcolor": "0.15",
                     "xtick.color": "0.35", "ytick.color": "0.35"})
tag = "PROVISIONAL" + (" (quick grid)" if QUICK else "")

# map
fig, ax = plt.subplots(figsize=(8.2, 4.6))
grid_Z = np.full((220, 120), np.nan)
grid_Z[ref["col"], ref["row"]] = ref["transfer_ohm"] * 1e-6 * 1e3     # mV per uA
im = ax.imshow(grid_Z.T, origin="lower", cmap="Blues",
               norm=LogNorm(vmin=np.nanmin(grid_Z), vmax=np.nanpercentile(grid_Z, 99.99)),
               extent=[-1.925, 1.925, -1.05, 1.05], aspect="equal", interpolation="nearest")
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"Potential at all 26,400 electrodes per µA delivered\n"
             f"return: top plate; {tag}", fontsize=10)
fig.colorbar(im, ax=ax, label="mV per µA (log scale)", shrink=0.85)
fig.tight_layout(); fig.savefig(FIG / "array_map.png", dpi=200); plt.close(fig)

# fall-off
fig, ax = plt.subplots(figsize=(7, 4.6))
for ret in RETURNS:
    r = fine[ret]
    sel = r["row"] == r["params"]["stim_col_row"][1]
    sel &= r["col"] > r["params"]["stim_col_row"][0]
    ax.loglog(r["distance_m"][sel] / um, r["transfer_ohm"][sel] * 1e-3, lw=2,
              color=COLORS[ret], label=f"return: {ret.replace('_', ' ')}")
d = np.geomspace(17.5, 2000, 200)
ax.loglog(d, 1 / (2 * np.pi * p["sigma_S_per_m"] * d * um) * 1e-3, ls="--", color="0.45",
          lw=1.2, label="ideal point source, I / (2πσr)")
ax.set_xlabel("distance from stimulating pad (µm)")
ax.set_ylabel("mV per µA")
ax.set_title(f"Fall-off along the stimulating row; {tag}", fontsize=10)
ax.grid(True, which="both", color="0.92", lw=0.5)
ax.legend(frameon=False, fontsize=8.5)
fig.tight_layout(); fig.savefig(FIG / "array_falloff.png", dpi=200); plt.close(fig)

# artifact time course
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
t = np.arange(-50e-6, 2 * PHASE + 150e-6, 0.05e-6)
steps = biphasic(A0, PHASE)
for ax, C in zip(axs, C_range):
    z1k = 1 / (2 * np.pi * 1e3 * C)
    label = "high-impedance end" if z1k >= max(chip["z_1kHz"]) * 0.999 else "low-impedance end"
    i_t = circuit_current(t, steps, R_used, C)
    for k, shade in [(1, "#1f4e8c"), (5, "#5b8fd0"), (20, "#a9c6ea")]:
        idx = probe_index(ref, k)
        phi = ref["transfer_ohm"][idx] * i_t * 1e3
        ax.plot(t * 1e6, phi, color=shade, lw=1.4, label=f"{k} pitch{'es' if k > 1 else ''} "
                f"({ref['distance_m'][idx]/um:.0f} µm)")
        tb, pb = sample_average(t - t[0], phi)
        ax.step((tb + t[0]) * 1e6, pb, where="mid", color=shade, lw=0.9, ls=":")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.set_title(f"{label}: |Z| {z1k/1e3:.0f} kΩ at 1 kHz, τ "
                 f"{R_used*C*1e6:.1f} µs", fontsize=9.5)
    ax.set_xlabel("time from pulse onset (µs)")
axs[0].set_ylabel("artifact (mV)")
axs[0].legend(frameon=False, fontsize=8, title="solid: true signal\ndotted: 50 µs averages",
              title_fontsize=8)
fig.suptitle(f"Predicted artifact, 500 mV biphasic, 200 µs/phase, ideal source; {tag}",
             fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "array_artifact.png", dpi=200); plt.close(fig)

# --------------------------------------------------------------------------
print(f"\n=== summary ({tag}) ===")
for ret in RETURNS:
    e = summary["per_return"][ret]
    R = e.get("R_solution_extrapolated_ohm", e["R_solution_ohm"][-1])
    g = e.get("R_solution_gci")
    print(f"{ret:10s} R(pad->return) {R/1e3:7.3f} kOhm" + (f"  GCI {g*100:.2f}%" if g is not None else ""))
print(f"\n{'pitches':>7} {'dist (um)':>9} " + " ".join(f"{r:>12}" for r in RETURNS) + "   spread")
for j, k in enumerate(PROBE_PITCHES):
    vals = [summary["per_return"][r]["probes"][j]["Z_ohm_coarse_to_fine"][-1] for r in RETURNS]
    print(f"{k:7d} {summary['per_return']['top_plate']['probes'][j]['distance_um']:9.1f} "
          + " ".join(f"{v:9.1f} Ohm" for v in vals) + f"   {spread[j]['spread_percent']:5.1f}%")
if not QUICK:
    print("\ngrid uncertainty at each probe (GCI if the grids converge steadily, "
          "otherwise the spread across the three grids):")
    print(f"{'pitches':>7} " + " ".join(f"{r:>18}" for r in RETURNS))
    for j, k in enumerate(PROBE_PITCHES):
        cells = []
        for ret in RETURNS:
            row = summary["per_return"][ret]["probes"][j]
            cells.append(f"{row['grid_uncertainty']*100:6.2f}% "
                         + ("(GCI)   " if row["gci_reliable"] else "(spread)"))
        print(f"{k:7d} " + " ".join(f"{c:>18}" for c in cells))
ip = interface["by_return"]["top_plate"]["pulses"]
print("\ninterface (top plate), 500 mV:")
for q in ip:
    if q["amplitude_V"] == 0.5:
        print(f"  C {q['C_dl_F']*1e9:.3f} nF: tau {q['tau_s']*1e6:.1f} us, peak current "
              f"{q['peak_current_A']*1e6:.1f} uA, charge/phase {q['charge_phase1_C']*1e12:.0f} pC")
print(f"\ntotal time {time.perf_counter() - t_all:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
