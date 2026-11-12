"""Research plan step 18 / section 5.6: which electrodes of the WHOLE array
(26,400) saturate their amplifiers after a stimulation pulse, and for how long.

Saturation is a property of the instrument, not of any cleaning method: no
method can recover a clipped sample. So this map is computed first and bounds
everything in Aim 2.

Model: artifact_k(t) = Z_k * i(t) at every electrode (Z from the solver, i
from the interface circuit, 500 mV biphasic, 200 us per phase). A 20 kHz
sample (50 us) is unusable if the instantaneous input exceeded the amplifier
range anywhere inside it, or within the assumed RECOVERY time after that.
Both the input range and the recovery time are undocumented, so both are
swept (families of curves, as the plan states):
    input range  +-1, 3, 10, 30 mV
    recovery     0.1, 0.5, 2 ms after the last clipped sample
    return path  top plate (provisional), well wall, periphery ring
    |Z| 1 kHz    300 and 600 kOhm (interface time constant)
The ideal voltage source makes this an UPPER bound on saturation.

Usage:
    python runs/saturation_map.py           (~3 min, ~4 GB)
    python runs/saturation_map.py --quick
Outputs: results/runs/saturation[_quick]/summary.json,
         figures/runs/saturation[_quick]/usable_fraction.png and saturation_map.png
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
from maxone import transfer_run, PITCH, N_COLS, N_ROWS, PROVISIONAL, CHIPS, um, mm
from interface import capacitance_from_impedance, biphasic, circuit_current

QUICK = "--quick" in sys.argv
label = "saturation_quick" if QUICK else "saturation"
OUT = ROOT / "results" / "runs" / label
FIG = ROOT / "figures" / "runs" / label
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
GRID = dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK else \
       dict(n_pad=8, growth=1.10, h_array=50 * um, h_far=0.5 * mm)

AMP, PHASE = 0.5, 200e-6
RANGES_mV = [1, 3, 10, 30]
RECOVERY_ms = [0.1, 0.5, 2.0]
RETURNS = ["top_plate", "well_wall", "periphery"]
Z1K = CHIPS["MaxOne"]["z_1kHz"]
TS = 50e-6
T_END = 4e-3
n_s = int(round(T_END / TS))
SUB = 50                                   # 1 us resolution inside each sample
t_fine = (np.arange(n_s * SUB) + 0.5) * TS / SUB
t_samp = np.arange(n_s) * TS               # sample k covers [k TS, (k+1) TS)

t0 = time.time()
fields = {r: transfer_run({"return": r}, verbose=False, **GRID) for r in RETURNS}
print(f"fields solved ({time.time()-t0:.0f} s)")

def unusable_until(Z, R_sol, z1k, rng_V, rec_s):
    """Per electrode: time (s) after onset until the LAST unusable sample ends
    (0 if never saturated), and the per-sample unusable mask."""
    i = circuit_current(t_fine, biphasic(AMP, PHASE), R_sol, capacitance_from_impedance(z1k))
    peak_in_sample = np.abs(i).reshape(n_s, SUB).max(1)          # A
    clipped = np.abs(Z)[:, None] * peak_in_sample[None, :] > rng_V
    last = np.where(clipped.any(1), n_s - 1 - np.argmax(clipped[:, ::-1], axis=1), -1)
    until = np.where(last >= 0, (last + 1) * TS + rec_s, 0.0)
    return until

summary = {"quick": QUICK, "grid": GRID, "amplitude_V": AMP, "phase_s": PHASE,
           "ranges_mV": RANGES_mV, "recovery_ms": RECOVERY_ms, "cases": {}}
times = np.arange(0, 3.01e-3, 50e-6)
fig, axs = plt.subplots(1, len(RETURNS), figsize=(15, 4.4), sharey=True, layout="constrained")
for ax, ret in zip(axs, RETURNS):
    F = fields[ret]
    stim = F["distance_m"] == 0
    Z = F["transfer_ohm"][~stim]
    dist = F["distance_m"][~stim]
    for rng_mV, c in zip(RANGES_mV, ["#c2410c", "#eb6834", "#2a78d6", "#1baf7a"]):
        for rec, ls in zip(RECOVERY_ms, ["-", "--", ":"]):
            for z1k in Z1K:
                until = unusable_until(Z, F["R_solution_ohm"], z1k, rng_mV * 1e-3, rec * 1e-3)
                key = f"{ret} | +-{rng_mV} mV | recovery {rec} ms | {z1k/1e3:.0f} kOhm"
                frac = [(until <= t).mean() for t in times]
                ever = until > 0
                summary["cases"][key] = {
                    "fraction_ever_saturated": float(ever.mean()),
                    "n_ever_saturated": int(ever.sum()),
                    "max_distance_saturated_um": float(dist[ever].max() / um) if ever.any() else 0.0,
                    "usable_fraction_vs_time": dict(zip([f"{t*1e3:.2f}" for t in times], map(float, frac)))}
                if z1k == Z1K[0]:
                    ax.plot(times * 1e3, np.array(frac) * 100, ls, color=c, lw=1.6,
                            label=f"+-{rng_mV} mV" if rec == RECOVERY_ms[0] else None)
    ax.set_title(f"return: {ret.replace('_', ' ')}", fontsize=10)
    ax.set_xlabel("time after pulse onset (ms)")
    ax.grid(True, color="0.92", lw=0.5)
axs[0].set_ylabel("usable electrodes (% of 26,399)")
axs[0].legend(frameon=False, fontsize=8, title="input range (line style:\nrecovery 0.1 / 0.5 / 2 ms)",
              title_fontsize=8)
fig.suptitle("Step 18: fraction of the array that can record, after a 500 mV pulse (|Z| 300 kOhm; "
             "ideal source = upper bound on saturation)" + (" -- QUICK" if QUICK else ""))
fig.savefig(FIG / "usable_fraction.png", dpi=200)

# map: saturation duration at +-3 mV, 0.5 ms recovery, top plate
F = fields["top_plate"]
until = unusable_until(F["transfer_ohm"], F["R_solution_ohm"], Z1K[0], 3e-3, 0.5e-3)
img = np.zeros((N_ROWS, N_COLS)); img[F["row"], F["col"]] = until * 1e3
c0, r0 = PROVISIONAL["stim_col_row"]
fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
im = ax.imshow(img, origin="lower", cmap="magma", extent=[-0.5, N_COLS - 0.5, -0.5, N_ROWS - 0.5])
ax.plot(c0, r0, "c+", ms=10)
fig.colorbar(im, ax=ax, label="unusable until (ms after onset)")
ax.set_xlabel("column"); ax.set_ylabel("row")
ax.set_title("Saturation map: +-3 mV range, 0.5 ms recovery, top-plate return, 300 kOhm")
fig.savefig(FIG / "saturation_map.png", dpi=200)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"{'case':58s} {'ever sat.':>9s} {'farthest':>9s}")
for k, v in summary["cases"].items():
    if "300 kOhm" in k and "0.5 ms" in k:
        print(f"{k:58s} {v['fraction_ever_saturated']*100:8.2f}% {v['max_distance_saturated_um']:7.0f} um")
print(f"total {time.time()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
