"""Aim 2: can a physics-predicted artifact be subtracted to recover spikes that
blanking throws away?

Synthetic ground truth -- every spike's time is known. Two "true" worlds,
both with conductivity 10% higher than the model assumes and interface
capacitance 0.40 nF (inside MaxWell's range for MaxOne platinum black):
  A. well-wall return, ideal stimulator (differs from the template's
     top-plate pattern almost exactly by a uniform offset -- the easy case)
  B. periphery-ring return (a spatially different pattern) and a stimulator
     with 20 kOhm output resistance (current decays ~2x slower than the
     template assumes) -- the hard case
  - physics template: built from the TOP-PLATE return at 1.5 S/m (the
    provisional model), so its spatial pattern is deliberately a little wrong;
    fitted per trial: overall scale, a uniform offset, and the interface
    time constant (3 numbers for all ~600 electrodes). Also run without the
    offset, and with 4 more smooth harmonic shapes (x, y, x^2-y^2, xy) that
    describe the potential a distant source adds across the array.
  - physics shape + onset gains: the physics TIME course (fitted tau) with
    each electrode's size measured from the first 100 us of the pulse,
    averaged over trials -- before any neuron can respond, so averaging is
    safe (saturated electrodes keep the solver's value).
    For when the return path is not known.
  - 60 trials of a 500 mV biphasic pulse, 200 us per phase
  - background spikes (random, 20 Hz per electrode) and EVOKED spikes (on 40%
    of electrodes, at a fixed delay 0.15-2.5 ms after the pulse, +-30 us
    jitter, in 60% of trials) -- evoked spikes are the neuroscience target
  - spike amplitude 50-200 uV; noise 5 uV rms; amplifier clips at +-3 mV
    (input range not documented -- also run at 1 and 10 mV) and needs 2
    samples (100 us) to recover after clipping
Electrodes: all within 5 pitches, the stimulating row out to 120 pitches each
side, plus 300 random electrodes across the array.

Methods: blanking (1 ms), average-over-trials template, physics template, and
two references: perfect template (the true artifact, an upper bound) and no
artifact at all.

    python runs/aim2_recovery.py            full (~5-10 min)
    python runs/aim2_recovery.py --quick    coarse field, fewer trials

Outputs: results/runs/aim2[_quick]/summary.json, figures/runs/aim2[_quick]/*.png
All settings are assumptions until the lab's amplifier details are known.
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import transfer_run, PITCH, N_COLS, N_ROWS, um, mm
from interface import capacitance_from_impedance
from recovery import (spike_shape, record, noise_level, detect, fit_physics,
                      harmonic_basis, unit_current_fine, unit_current_binned,
                      electrode_gains, bin_average, TS)

QUICK = "--quick" in sys.argv
NAME = "aim2_quick" if QUICK else "aim2"
OUT = ROOT / "results" / "runs" / NAME
FIG = ROOT / "figures" / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20261110)

GRID = (dict(n_pad=4, growth=1.2, h_array=100 * um, h_far=1.0 * mm) if QUICK
        else dict(n_pad=6, growth=1 + 0.1 * 2 ** 0.5, h_array=50 * um * 2 ** 0.5,
                  h_far=0.5 * mm * 2 ** 0.5))
N_TRIALS = 20 if QUICK else 60
SUB = 20                                    # fine steps per 50 us sample (2.5 us)
T0, T1 = -1.0e-3, 5.0e-3                    # window around pulse onset (s)
AMP = 0.5                                   # V
SIGMA_TRUE_FACTOR = 1.1                     # true conductivity / assumed
C_TRUE = 0.40e-9
NOISE = 5e-6
SATS = [3e-3] if QUICK else [1e-3, 3e-3, 10e-3]
RECOVER = 2
T_BLANK = 1.0e-3
LAT_BINS_MS = [(0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)]
DIST_BANDS = [(1, 5), (5, 20), (20, 130)]

SCENARIOS = {"A: well-wall return": {"ret": "well_wall", "R_buf": 0.0},
             "B: periphery return + 20 kOhm stimulator": {"ret": "periphery", "R_buf": 20e3}}
t_start = time.perf_counter()
fields = {}
for ret in ["top_plate", "well_wall", "periphery"]:
    r = transfer_run({"return": ret}, verbose=False, **GRID)
    fields[ret] = r
    print(f"field '{ret}': {r['grid']['unknowns']:,} unknowns, R {r['R_solution_ohm']/1e3:.2f} kOhm",
          flush=True)

ref = fields["top_plate"]
c0, r0 = ref["params"]["stim_col_row"]
row_sel = np.flatnonzero((ref["row"] == r0) & (ref["col"] != c0) & (np.abs(ref["col"] - c0) <= 120))
others = np.setdiff1d(np.flatnonzero(ref["col"] != c0), row_sel)
# every electrode within 5 pitches (so the near band is well sampled), the
# stimulating row, and random electrodes across the array
near = np.flatnonzero((np.abs(ref["col"] - c0) <= 5) & (np.abs(ref["row"] - r0) <= 5)
                      & ~((ref["col"] == c0) & (ref["row"] == r0)))
others = np.setdiff1d(others, near)
rand_sel = rng.choice(others, 100 if QUICK else 300, replace=False)
el = np.unique(np.concatenate([near, row_sel, rand_sel]))
n_el = len(el)
dist_p = ref["distance_m"][el] / PITCH
Zhat = ref["transfer_ohm"][el]
H = harmonic_basis(ref["x"][el], ref["y"][el], order=2)[:, 1:]      # x, y, x^2-y^2, xy
R_hat = ref["R_solution_ohm"]
print(f"{n_el} electrodes, {dist_p.min():.0f}-{dist_p.max():.0f} pitches from the pad")

# time axes
n_samp = int(round((T1 - T0) / TS))
t_fine = T0 + (np.arange(n_samp * SUB) + 0.5) * (TS / SUB)
t_samp = T0 + (np.arange(n_samp) + 0.5) * TS
baseline = t_samp < 0
# the fitted capacitance only sets the time constant (with R_hat), so search widely
C_grid = np.geomspace(0.1e-9, 1.5e-9, 25)

# spikes
evoked_on = rng.random(n_el) < 0.4
evoked_lat = rng.uniform(0.15e-3, 2.5e-3, n_el)
amp_el = rng.uniform(50e-6, 200e-6, n_el)

def make_spikes():
    sp = np.zeros((n_el, len(t_fine)))
    truth = []                               # (electrode, trough time s, kind)
    for e in range(n_el):
        times = []
        n_bg = rng.poisson(20 * (T1 - T0))
        times += [(t, "background") for t in rng.uniform(T0 + 0.5e-3, T1 - 1e-3, n_bg)]
        if evoked_on[e] and rng.random() < 0.6:
            times.append((evoked_lat[e] + rng.normal(0, 30e-6), "evoked"))
        for t, kind in times:
            sp[e] += amp_el[e] * spike_shape((t_fine - t) * 1e3)
            truth.append((e, t, kind))
    return sp, truth

trials = [make_spikes() for _ in range(N_TRIALS)]

def score(dets_by_trial, truth_by_trial):
    rows = []
    fp = 0
    for dets, truth in zip(dets_by_trial, truth_by_trial):
        det = {}
        for e, s in dets:
            det.setdefault(e, []).append(s)
        used = set()
        for e, t, kind in truth:
            s_true = int(np.floor((t - T0) / TS))
            cand = [s for s in det.get(e, []) if abs(s - s_true) <= 2 and (e, s) not in used]
            found = bool(cand)
            if found:
                used.add((e, cand[0]))
            rows.append((e, t, kind, found))
        fp += sum(1 for e, s in dets if (e, s) not in used)
    return rows, fp

def summarise(rows, fp, n_trials):
    out = {"false_positives_per_electrode_second": fp / (n_el * n_trials * (T1 - T0))}
    for kind in ["evoked", "background"]:
        for a, b in LAT_BINS_MS:
            for d0, d1 in DIST_BANDS:
                sel = [f for e, t, k, f in rows if k == kind and a <= t * 1e3 < b
                       and d0 <= dist_p[e] < d1]
                out[f"{kind}|{a}-{b} ms|{d0}-{d1} p"] = (float(np.mean(sel)) if sel else None, len(sel))
            sel = [f for e, t, k, f in rows if k == kind and a <= t * 1e3 < b]
            out[f"{kind}|{a}-{b} ms|all"] = (float(np.mean(sel)) if sel else None, len(sel))
    return out

summary = {"settings": {"trials": N_TRIALS, "amplitude_V": AMP, "C_true_F": C_TRUE,
                        "sigma_true_over_assumed": SIGMA_TRUE_FACTOR, "noise_V": NOISE,
                        "recover_samples": RECOVER, "blank_s": T_BLANK, "scenarios": SCENARIOS,
                        "template_return": "top_plate", "electrodes": n_el, "grid": GRID,
                        "status": "synthetic test; amplifier settings assumed"},
           "by_scenario": {}}

for scen, sc in SCENARIOS.items():
  Z_true = fields[sc["ret"]]["transfer_ohm"][el] / SIGMA_TRUE_FACTOR
  R_true = fields[sc["ret"]]["R_solution_ohm"] / SIGMA_TRUE_FACTOR
  i_true = AMP * unit_current_fine(R_true, C_TRUE, t_fine - T0, -T0, R_buf=sc["R_buf"])
  art_true = Z_true[:, None] * i_true[None, :]
  summary["by_scenario"][scen] = {}
  print(f"\n##### scenario {scen} #####")
  for sat in SATS:
      t1 = time.perf_counter()
      rec = {"no artifact": [], "blanking": [], "average template": [],
             "physics, no offset": [], "physics template": [], "physics + harmonics": [],
             "physics shape + onset gains": [], "perfect template": []}
      ys, sats_, raw_noart = [], [], []
      for sp, truth in trials:
          y, s = record(art_true, sp, NOISE, sat, SUB, RECOVER, rng)
          y0, s0 = record(np.zeros_like(art_true), sp, NOISE, sat, SUB, 0, rng)
          ys.append(y); sats_.append(s); raw_noart.append((y0, s0))
      sigma = noise_level(np.concatenate([y[:, baseline] for y in ys], axis=1),
                          np.ones(np.sum(baseline) * len(ys), dtype=bool))
      avg = np.mean(ys, axis=0)
      true_b = bin_average(art_true, SUB)
      fit_info = []
      for (sp, truth), y, s, (y0, s0) in zip(trials, ys, sats_, raw_noart):
          usable = ~s
          rec["no artifact"].append(detect(y0, ~s0, sigma))
          blank = usable & ~((t_samp >= 0) & (t_samp < T_BLANK))[None, :]
          rec["blanking"].append(detect(y, blank, sigma))
          rec["average template"].append(detect(y - avg, usable, sigma))
          rec["perfect template"].append(detect(y - true_b, usable, sigma))
          win = (t_samp > -0.1e-3) & (t_samp < 1.5e-3)
          tmpl0, _, _ = fit_physics(y, usable, Zhat, R_hat, C_grid, t_fine - T0, -T0, SUB,
                                    window=win, offset=False)
          rec["physics, no offset"].append(detect(y - tmpl0, usable, sigma))
          tmpl, coef, C_fit = fit_physics(y, usable, Zhat, R_hat, C_grid, t_fine - T0, -T0, SUB,
                                          window=win)
          # charge per phase relative to the truth: (alpha * C) vs (AMP * C_true * R-ratio)
          fit_info.append((coef[0] * C_fit / (AMP * C_TRUE), C_fit, coef[1], coef[0]))
          rec["physics template"].append(detect(y - tmpl, usable, sigma))
          tmplh, _, _ = fit_physics(y, usable, Zhat, R_hat, C_grid, t_fine - T0, -T0, SUB,
                                    window=win, extra=H)
          rec["physics + harmonics"].append(detect(y - tmplh, usable, sigma))
      # physics time course + per-electrode gains from the onset edge (first 100 us),
      # averaged over trials: nothing evoked can be there yet, so averaging is safe
      onset = (t_samp > -0.1e-3) & (t_samp < 0.1e-3)
      C_med = float(np.median([f[1] for f in fit_info]))
      ib = unit_current_binned(R_hat, C_med, t_fine - T0, -T0, SUB)
      U = ~np.array(sats_)
      all_ok = U.all(axis=0)                                  # usable in every trial
      y_mean = np.mean(ys, axis=0)
      fb = np.median([f[3] for f in fit_info]) * Zhat + np.median([f[2] for f in fit_info])
      g, ok = electrode_gains(y_mean, all_ok, ib, onset, fb)
      n_meas = [ok.mean()]
      for (sp, truth), y, s in zip(trials, ys, sats_):
          rec["physics shape + onset gains"].append(detect(y - g[:, None] * ib[None, :], ~s, sigma))
      res = {}
      for m, dets in rec.items():
          rows, fp = score(dets, [tr for _, tr in trials])
          res[m] = summarise(rows, fp, N_TRIALS)
      sat_frac = float(np.mean([s[:, (t_samp >= 0) & (t_samp < 1e-3)].mean() for s in sats_]))
      fi = np.array(fit_info)
      res["fit"] = {"fitted_charge_over_true_mean": float(np.mean(fi[:, 0])),
                    "C_fit_mean_nF": float(np.mean(fi[:, 1]) * 1e9), "C_true_nF": C_TRUE * 1e9,
                    "offset_beta_mean": float(np.mean(fi[:, 2])),
                    "electrodes_with_measured_onset_gain": float(np.mean(n_meas)),
                    "saturated_fraction_first_ms": sat_frac}
      summary["by_scenario"][scen][f"{sat*1e3:g} mV"] = res
      print(f"\n=== amplifier range +-{sat*1e3:g} mV  ({time.perf_counter()-t1:.0f} s) ===")
      print(f"saturated samples in the first ms: {100*sat_frac:.1f}%;  physics fit: "
            f"C {np.mean(fi[:,1])*1e9:.3f} nF (true {C_TRUE*1e9:.2f}), uniform offset "
            f"{np.mean(fi[:,2]):.1f} (V per A x volts)")
      print(f"evoked spikes found (fraction), all distances:")
      print(f"{'method':28s} " + " ".join(f"{f'{a}-{b} ms':>11}" for a, b in LAT_BINS_MS) + "   false/el/s")
      for m in rec:
          cells = []
          for a, b in LAT_BINS_MS:
              v, n = res[m][f"evoked|{a}-{b} ms|all"]
              cells.append(f"{v:11.2f}" if v is not None else f"{'-':>11}")
          print(f"{m:28s} " + " ".join(cells) + f"   {res[m]['false_positives_per_electrode_second']:9.2f}")
      print("evoked spikes found 0-0.5 ms after the pulse, by distance:")
      for d0, d1 in DIST_BANDS:
          cells = []
          for m in rec:
              ff, nn = [], 0
              for a, b in LAT_BINS_MS[:2]:
                  v, n = res[m][f"evoked|{a}-{b} ms|{d0}-{d1} p"]
                  if v is not None:
                      ff.append(v * n); nn += n
              cells.append(f"{m}: {sum(ff)/nn:.2f}" if nn else f"{m}: -")
          print(f"  {d0:3d}-{d1:3d} pitches  " + " | ".join(cells))

(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

# ---------------- figure: evoked recovery vs latency, +-3 mV range ----------------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "0.4", "axes.labelcolor": "0.15",
                     "xtick.color": "0.35", "ytick.color": "0.35"})
key = "3 mV"
colors = {"blanking": "#eb6834", "average template": "#1baf7a", "physics template": "#2a78d6",
          "perfect template": "0.55", "no artifact": "0.75", "physics, no offset": "#2a78d6",
          "physics + harmonics": "#7b4fd0", "physics shape + onset gains": "#111111"}
styles = {"perfect template": "--", "no artifact": ":", "physics, no offset": ":"}
xc = [np.mean(b) for b in LAT_BINS_MS]
for si, scen in enumerate(SCENARIOS):
    res = summary["by_scenario"][scen][key]
    fig, axs = plt.subplots(1, len(DIST_BANDS) + 1, figsize=(17, 4.4), layout="constrained",
                            gridspec_kw={"width_ratios": [1] * len(DIST_BANDS) + [0.75]})
    for ax, (d0, d1) in zip(axs[:-1], DIST_BANDS):
        for m in ["no artifact", "perfect template", "blanking", "average template",
                  "physics template", "physics shape + onset gains"]:
            v = [res[m][f"evoked|{a}-{b} ms|{d0}-{d1} p"][0] for a, b in LAT_BINS_MS]
            v = [np.nan if q is None else q for q in v]
            ax.plot(xc, v, styles.get(m, "-"), marker="o", ms=3.5, lw=1.8, color=colors[m], label=m)
        ax.set_title(f"electrodes {d0}-{d1} pitches from the pad", fontsize=9.5)
        ax.set_xlabel("spike time after pulse onset (ms)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, color="0.92", lw=0.5)
    for ax in axs[1:-1]:
        ax.set_yticklabels([])
    axs[0].set_ylabel("fraction of evoked spikes found")
    axs[0].legend(frameon=False, fontsize=8)
    # false detections: what separates the methods that all find the spikes
    ms = ["perfect template", "blanking", "average template", "physics template",
          "physics shape + onset gains"]
    fp = [res[m]["false_positives_per_electrode_second"] for m in ms]
    ax = axs[-1]
    ax.barh(range(len(ms)), fp, color=[colors[m] for m in ms], height=0.6)
    for k, v in enumerate(fp):
        ax.text(v * 1.15, k, f"{v:.1f}", va="center", fontsize=8.5, color="0.2")
    ax.set_xscale("log"); ax.set_xlim(0.1, 1000)
    ax.set_yticks(range(len(ms))); ax.set_yticklabels(ms, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("per electrode per second")
    ax.set_title("false detections (lower is better)", fontsize=9.5)
    ax.grid(True, axis="x", color="0.92", lw=0.5)
    fig.suptitle(f"Aim 2, scenario {scen}: recovering evoked spikes (synthetic, amplifier range "
                 f"±{key}){' — quick' if QUICK else ''}", fontsize=10)
    fig.savefig(FIG / f"evoked_recovery_{'AB'[si]}.png", dpi=200); plt.close(fig)
print(f"\ntotal {time.perf_counter()-t_start:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
