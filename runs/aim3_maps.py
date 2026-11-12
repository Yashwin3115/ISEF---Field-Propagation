"""Aim 3, research plan steps 37-42 and section 5.10: activation maps,
resolvability, spatial misattribution, sweeps, and the Ronchi et al. check.

NEURON-LIKE AXON (for the maps). A straight unmyelinated axon, 1 um diameter,
1 mm long, 1 um above the chip, starting at a soma end (sealed) at (x0, y0):
  0-5 um      hillock, Na 8000 pS/um^2
  5-45 um     axon initial segment (AIS), Na 30000 pS/um^2
  45 um-1 mm  axon, Na 8000 pS/um^2
(Mainen-Sejnowski 37 C channels, as in aim3_activation.py; the resting state
is settled before every run because the AIS rests at a different voltage.)
"Fires" = an action potential reaches 500 um down the axon.

MAP (steps 38, 42). Soma positions on a grid around the pad (step 5 um,
+-40 um), axon pointing along +x (0 deg) and +y (90 deg). Threshold at every
position with the solved field and with the point source I / (2 pi sigma r).
Activated region at 400/500/600 mV = positions with threshold <= amplitude.
Reported: Jaccard overlap of the full and point regions, centroid displacement
(um and electrode pitches).

RESOLVABILITY (step 41). The full-field threshold is computed on three grids
(refinement levels as in the blind prediction); its uncertainty (GCI, or the
spread if the grids do not converge steadily) plus 0.5% for the cable
discretization is the uncertainty of each threshold. A position where the
two field models disagree about activation counts as a RESOLVED difference
only if the full-field threshold is farther from the amplitude than its
uncertainty (the point formula is exact for what it assumes). If no
disagreement is resolved, the comparison is reported as "not resolvable at
the achieved grid resolution", as the plan requires.

SWEEPS (step 37). Uniform axon (Na 8000, as in aim3_activation.py), centred
past the pad at sideways offsets 0-40 um: diameter 0.5 / 1 / 2 um x phase
width 100 / 200 / 400 us. Threshold-vs-distance exponent fitted over
10-30 um (section 5.10) and compared with 2 (the classical value, derived for
myelinated fibres far from a point source).

RONCHI CHECK (step 40). Same neuron-like axon, pad under the AIS vs pad under
the axon shaft 300 um from the soma: Ronchi et al. 2019 report that the most
efficient stimulation site is near the AIS. Reported as agreement or not.

    python runs/aim3_maps.py            (~35-45 min, ~4 GB)
    python runs/aim3_maps.py --quick    (coarse, ~3 min; not for results)
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from maxone import transfer_run, electrode_xy, CHIPS, PROVISIONAL, PITCH, um, mm
from interface import capacitance_from_impedance, biphasic, circuit_current
from neuron import Axon, MainenMembrane, SegmentedMembrane, settle, straight_axon_points, thresholds
from uncertainty import LEVELS, grid_settings, uncertainty_of

QUICK = "--quick" in sys.argv
NAME = "aim3_maps_quick" if QUICK else "aim3_maps"
OUT = ROOT / "results" / "runs" / NAME
FIG = ROOT / "figures" / "runs" / NAME
OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

LEV = [4.0, 2.0 * 2 ** 0.5, 2.0] if QUICK else LEVELS
DT, T_END, T_PULSE = 0.001, 2.0, 0.1            # ms
PHASE = 200e-6
Z0 = 1 * um
L_AX, N_AX = 1 * mm, 500                        # 2 um compartments
STEP = 10 * um if QUICK else 5 * um
SPAN = 40 * um
AMPS = [0.4, 0.5, 0.6]
AMP_MAX = 3.0
ITER = 9 if QUICK else 11
chip = CHIPS[PROVISIONAL["chip"]]
Z1K = chip["z_1kHz"][0]                          # 300 kOhm end
sigma = PROVISIONAL["sigma_S_per_m"]
sx, sy = electrode_xy(*PROVISIONAL["stim_col_row"])
t0 = time.perf_counter()

# ------------------------------------------------------------------ fields
fields, Ns, Rsol = [], [], []
half = L_AX + 0.1 * mm
for f in LEV:
    r = transfer_run({}, verbose=False, **grid_settings(f))
    x, y, z, phi = r["x_grid"], r["y_grid"], r["z_grid"], r["phi_per_A"]
    def widen(m):                      # the box plus two nodes on each side
        idx = np.flatnonzero(m)
        out = np.zeros_like(m)
        out[max(idx[0] - 2, 0):min(idx[-1] + 3, len(m))] = True
        return out
    bx = widen((x > sx - half) & (x < sx + half)); by = widen((y > sy - half) & (y < sy + half))
    bz = widen(z < 40 * um)
    fields.append(RegularGridInterpolator((x[bx], y[by], z[bz]), phi[np.ix_(bx, by, bz)],
                                          method="cubic", bounds_error=True))
    Ns.append(r["grid"]["unknowns"]); Rsol.append(r["R_solution_ohm"])
    del r, phi
    print(f"field at grid level {f:.3g}: {Ns[-1]:,} unknowns ({time.perf_counter()-t0:.0f} s)", flush=True)
R_sol = Rsol[-1]

def unit_current(phase_s=PHASE):
    steps = int(round(T_END / DT)); sub = 20
    t = (np.arange(steps * sub) + 0.5) * (DT / sub) * 1e-3
    i = circuit_current(t, biphasic(1.0, phase_s, T_PULSE * 1e-3), R_sol, capacitance_from_impedance(Z1K))
    return i.reshape(steps, sub).mean(axis=1)

def point_field(pts):
    rr = np.sqrt((pts[:, 0] - sx) ** 2 + (pts[:, 1] - sy) ** 2 + pts[:, 2] ** 2)
    return 1 / (2 * np.pi * sigma * rr)

# ------------------------------------------------------------------ neuron-like axon
gna = np.full(N_AX, 8000.0)
s_mid = (np.arange(N_AX) + 0.5) * (L_AX / N_AX)
gna[(s_mid >= 5 * um) & (s_mid < 45 * um)] = 30000.0
def neuron_axon(d_cm=1e-4):
    ax = Axon(n=N_AX, dx_cm=L_AX / N_AX * 100, d_cm=d_cm, Ra_ohm_cm=150.0, dt_ms=DT,
              membrane=SegmentedMembrane(gna))
    drift = settle(ax, 20.0)
    return ax, drift
CHECK_N = [int(round(500 * um / (L_AX / N_AX)))]             # 500 um down the axon
iu = unit_current()

def soma_axons(starts_xy, angle_deg):
    th = np.deg2rad(angle_deg); u = np.array([np.cos(th), np.sin(th)])
    P = [straight_axon_points(np.array([*s0, Z0]), u, L_AX, N_AX, start=True)[0] for s0 in starts_xy]
    return P

g1 = np.arange(-SPAN, SPAN + 1e-12, STEP)
G1x, G1y = np.meshgrid(g1, g1, indexing="ij")
starts = np.column_stack([sx + G1x.ravel(), sy + G1y.ravel()])
results = {"settings": {"levels": LEV, "unknowns": Ns, "map_step_um": STEP / um, "span_um": SPAN / um,
                        "z0_um": Z0 / um, "impedance_kOhm": Z1K / 1e3, "amplitudes_V": AMPS,
                        "axon": "neuron-like: AIS 5-45 um at Na 30000, rest Na 8000, 1 um, 1 mm",
                        "quick": QUICK}}
ax_n, drift = neuron_axon()
print(f"neuron-like axon settled (residual drift {drift:.1e} mV/ms)")

maps = {}
for ang in (0, 90):
    P = soma_axons(starts, ang)
    th_full = []
    levels_to_run = range(len(LEV)) if ang == 0 else [len(LEV) - 1]
    for li in levels_to_run:
        Phi = np.column_stack([fields[li](p) for p in P])
        t1 = time.perf_counter()
        th_full.append(thresholds(ax_n, Phi, iu, T_END, CHECK_N, amp_max_V=AMP_MAX, iterations=ITER))
        print(f"  map {ang} deg, full field, level {LEV[li]:.3g}: {time.perf_counter()-t1:.0f} s", flush=True)
    Phi = np.column_stack([point_field(p) for p in P])
    th_pt = thresholds(ax_n, Phi, iu, T_END, CHECK_N, amp_max_V=AMP_MAX, iterations=ITER)
    print(f"  map {ang} deg, point field done ({time.perf_counter()-t0:.0f} s)", flush=True)
    U = np.full(len(starts), np.nan)
    if len(th_full) == 3:
        for k in range(len(starts)):
            u, _ = uncertainty_of([th_full[0][k], th_full[1][k], th_full[2][k]], Ns)
            U[k] = np.hypot(u, 0.005) if np.isfinite(u) else np.nan
    maps[ang] = {"full": th_full[-1], "point": th_pt, "U_rel": U,
                 "full_by_level": [t.tolist() for t in th_full]}

def region_stats(thF, thP, U, A):
    F = np.isfinite(thF) & (thF <= A)
    Pm = np.isfinite(thP) & (thP <= A)
    inter, union = (F & Pm).sum(), (F | Pm).sum()
    jac = inter / union if union else np.nan
    cF = starts[F].mean(0) if F.any() else np.array([np.nan, np.nan])
    cP = starts[Pm].mean(0) if Pm.any() else np.array([np.nan, np.nan])
    disp = float(np.hypot(*(cF - cP)))
    dis = F ^ Pm
    if np.all(np.isnan(U)):
        resolved = None
    else:
        # full-field classification certain if |thF - A| > U * thF (unfired: thF = NaN -> > 3 V, certain)
        cert = np.where(np.isfinite(thF), np.abs(thF - A) > U * thF, True)
        resolved = int((dis & cert).sum())
    return {"jaccard": float(jac), "n_full": int(F.sum()), "n_point": int(Pm.sum()),
            "centroid_displacement_um": disp / um, "centroid_displacement_pitches": disp / PITCH,
            "n_disagree": int(dis.sum()), "n_disagree_resolved": resolved,
            "verdict": ("no disagreement" if dis.sum() == 0 else
                        "difference resolved" if resolved else
                        "not resolvable at the achieved grid resolution" if resolved == 0 else
                        "resolvability not tested (single grid)")}

results["maps"] = {}
print("\nactivated region, full vs point field (soma positions within +-40 um)")
for ang, m in maps.items():
    for A in AMPS:
        st = region_stats(m["full"], m["point"], m["U_rel"], A)
        results["maps"][f"{ang} deg | {A*1e3:.0f} mV"] = st
        print(f"  {ang:2d} deg {A*1e3:.0f} mV: full {st['n_full']:3d} positions, point {st['n_point']:3d}; "
              f"Jaccard {st['jaccard']:.2f}; centroid shift {st['centroid_displacement_um']:.1f} um "
              f"({st['centroid_displacement_pitches']:.2f} pitches); disagree {st['n_disagree']}, "
              f"resolved {st['n_disagree_resolved']} -> {st['verdict']}")
    results["maps"][f"{ang} deg | thresholds"] = {"full_mV": (m["full"] * 1e3).tolist(),
                                                  "point_mV": (m["point"] * 1e3).tolist(),
                                                  "U_rel": m["U_rel"].tolist(),
                                                  "full_by_level_mV": (np.array(m["full_by_level"]) * 1e3).tolist()}
U0 = maps[0]["U_rel"]
if np.any(np.isfinite(U0)):
    fin = np.isfinite(U0) & np.isfinite(maps[0]["full"])
    print(f"threshold grid uncertainty (0 deg map): median {np.nanmedian(U0[fin])*100:.2f}%, "
          f"max {np.nanmax(U0[fin])*100:.2f}%")
    results["threshold_uncertainty_median"] = float(np.nanmedian(U0[fin]))

# ------------------------------------------------------------------ Ronchi check
ys_ = np.array([0, 5, 10, 15]) * um
ais_starts = np.column_stack([np.full(len(ys_), sx - 25 * um), sy + ys_])       # AIS centre over the pad
shaft_starts = np.column_stack([np.full(len(ys_), sx - 300 * um), sy + ys_])    # shaft 300 um from soma
P = soma_axons(np.vstack([ais_starts, shaft_starts]), 0)
Phi = np.column_stack([fields[-1](p) for p in P])
th = thresholds(ax_n, Phi, iu, T_END, CHECK_N, amp_max_V=AMP_MAX, iterations=ITER) * 1e3
results["ronchi_check"] = {"offsets_um": (ys_ / um).tolist(), "AIS_over_pad_mV": th[:4].tolist(),
                           "shaft_over_pad_mV": th[4:].tolist(),
                           "agrees_AIS_most_excitable": bool(np.nanmean(th[:4]) < np.nanmean(th[4:]))}
print("\nRonchi check (pad under the AIS vs under the shaft 300 um away), threshold mV:")
for yy, a, b in zip(ys_ / um, th[:4], th[4:]):
    print(f"  offset {yy:4.0f} um: AIS {a:7.1f}   shaft {b:7.1f}")
print(f"  -> AIS more excitable: {results['ronchi_check']['agrees_AIS_most_excitable']}")

# ------------------------------------------------------------------ sweeps
offs = np.array([0, 5, 10, 15, 20, 30, 40]) * um
DIAMS = [0.5, 1.0] if QUICK else [0.5, 1.0, 2.0]
WIDTHS = [200e-6] if QUICK else [100e-6, 200e-6, 400e-6]
sweep = {}
for d in DIAMS:
    ax_u = Axon(n=N_AX, dx_cm=L_AX / N_AX * 100, d_cm=d * 1e-4, Ra_ohm_cm=150.0, dt_ms=DT,
                membrane=MainenMembrane(8000.0))
    P = [straight_axon_points(np.array([sx, sy + o, Z0]), np.array([1.0, 0.0]), L_AX, N_AX)[0] for o in offs]
    Phi = np.column_stack([fields[-1](p) for p in P])
    chk = [N_AX // 2 - 150, N_AX // 2 + 150]
    for w in WIDTHS:
        th = thresholds(ax_u, Phi, unit_current(w), T_END, chk, amp_max_V=AMP_MAX, iterations=ITER) * 1e3
        sweep[f"{d} um | {w*1e6:.0f} us"] = th.tolist()
        print(f"  sweep d {d} um, {w*1e6:.0f} us/phase: " + " ".join(f"{v:7.0f}" for v in th), flush=True)
results["sweep"] = {"offsets_um": (offs / um).tolist(), "thresholds_mV": sweep}
th1 = np.array(sweep[f"1.0 um | 200 us"])
m = (offs >= 10 * um) & (offs <= 30 * um) & np.isfinite(th1)
expo = float(np.polyfit(np.log(offs[m]), np.log(th1[m]), 1)[0]) if m.sum() >= 2 else np.nan
results["distance_exponent_10_30um"] = expo
print(f"\nthreshold ~ distance^{expo:.2f} over 10-30 um (classical value 2, for myelinated fibres "
      f"far from a point source)")

# ------------------------------------------------------------------ figures
fig, axs = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
ext = [(-SPAN - STEP / 2) / um, (SPAN + STEP / 2) / um] * 2
for row, ang in enumerate((0, 90)):
    for col, (key, title) in enumerate([("full", "solved field"), ("point", "point source")]):
        img = (maps[ang][key] * 1e3).reshape(len(g1), len(g1)).T
        ax = axs[row, col]
        im = ax.imshow(np.where(np.isfinite(img), img, np.nan), origin="lower", extent=ext,
                       cmap="viridis_r", vmin=0, vmax=1500)
        ax.contour(g1 / um, g1 / um, np.nan_to_num(img, nan=1e4), levels=[400, 500, 600],
                   colors=["w", "0.8", "0.6"], linewidths=1)
        ax.add_patch(plt.Rectangle((-chip["pad_x"] / 2 / um, -chip["pad_y"] / 2 / um),
                                   chip["pad_x"] / um, chip["pad_y"] / um, fill=False, ec="r", lw=1))
        ax.set_title(f"{title}, axon toward {ang} deg: threshold (mV)", fontsize=9.5)
        ax.set_xlabel("soma x - pad x (um)"); ax.set_ylabel("soma y - pad y (um)")
        fig.colorbar(im, ax=ax, shrink=0.8)
    ax = axs[row, 2]
    for A, c in zip(AMPS, ["#1baf7a", "#2a78d6", "#eb6834"]):
        F = np.isfinite(maps[ang]["full"]) & (maps[ang]["full"] <= A)
        Pm = np.isfinite(maps[ang]["point"]) & (maps[ang]["point"] <= A)
        ax.scatter((starts[F, 0] - sx) / um, (starts[F, 1] - sy) / um, s=40, facecolors="none",
                   edgecolors=c, label=f"full <= {A*1e3:.0f} mV")
        ax.scatter((starts[Pm, 0] - sx) / um, (starts[Pm, 1] - sy) / um, s=8, color=c, marker="x")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[0], ext[1]); ax.set_aspect("equal")
    ax.set_title("activated somas: o full field, x point source", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7)
fig.suptitle("Aim 3 activation maps (neuron-like axon with AIS, 1 um above the chip, 300 kOhm); PROVISIONAL"
             + (" -- QUICK" if QUICK else ""))
fig.savefig(FIG / "activation_maps.png", dpi=180)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
for k, v in sweep.items():
    d, w = k.split(" | ")
    ls = {"100 us": ":", "200 us": "-", "400 us": "--"}.get(w, "-")
    c = {"0.5 um": "#eb6834", "1.0 um": "#2a78d6", "2.0 um": "#1baf7a"}.get(d, "k")
    axs[0].semilogy(offs / um, v, ls, color=c, marker="o", ms=3, label=f"{d}, {w}")
for A in AMPS:
    axs[0].axhline(A * 1e3, color="0.7", lw=0.8)
axs[0].set_xlabel("sideways offset (um)"); axs[0].set_ylabel("threshold (mV)")
axs[0].set_title("uniform axon: diameter (colour) and phase width (line)", fontsize=9.5)
axs[0].legend(frameon=False, fontsize=7, ncol=2)
rc = results["ronchi_check"]
xx = np.arange(4)
axs[1].bar(xx - 0.2, rc["AIS_over_pad_mV"], 0.4, label="pad under the AIS", color="#2a78d6")
axs[1].bar(xx + 0.2, rc["shaft_over_pad_mV"], 0.4, label="pad under the shaft (300 um)", color="0.6")
axs[1].set_xticks(xx); axs[1].set_xticklabels([f"{v:.0f} um" for v in rc["offsets_um"]])
axs[1].set_xlabel("sideways offset"); axs[1].set_ylabel("threshold (mV)")
axs[1].set_title("Ronchi et al. 2019 check: is the AIS the most excitable site?", fontsize=9.5)
axs[1].legend(frameon=False, fontsize=8)
fig.savefig(FIG / "sweeps_and_ronchi.png", dpi=180)
(OUT / "summary.json").write_text(json.dumps(results, indent=2, default=float))
print(f"\ntotal {time.perf_counter()-t0:.0f} s; wrote {OUT.relative_to(ROOT)}/ and {FIG.relative_to(ROOT)}/")
