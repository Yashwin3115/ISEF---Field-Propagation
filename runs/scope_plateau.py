"""One reading for the measurements CSV from oscilloscope trace files
(plan steps 53-54), with a noise floor that is REAL: random noise across the
traces plus the part of the 8-bit quantization that averaging cannot remove.

    python runs/scope_plateau.py data/phantom/raw/s1_pos03_200us_*.csv
    options:  --channel CH2      which column (default: the first channel)
              --vdiv 0.002       volts/div, if the scope averaged internally
              --bits 8 --divs 8  ADC bits and full-scale divisions (with --vdiv)

Reads any common scope CSV layout (see src/scope.py). Prints each file's
first-phase plateau (baseline subtracted, middle half of the phase), then the
line to paste into the measurements CSV: plateau_V, plateau_V_sd, n_traces,
noise_floor_V. Raw files are never modified.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scope import read_scope_csv, plateau, plateau_reading, lsb_from_settings

def opt(name, default=None, cast=float):
    if name in sys.argv:
        k = sys.argv.index(name)
        v = cast(sys.argv[k + 1]); del sys.argv[k:k + 2]
        return v
    return default
channel = opt("--channel", None, str)
vdiv, bits, divs = opt("--vdiv"), opt("--bits", 8, int), opt("--divs", 8.0)
files = sys.argv[1:]
if not files:
    sys.exit(__doc__)
traces = []
for p in files:
    d = read_scope_csv(p)
    name = channel or next(iter(d["channels"]))
    if name not in d["channels"]:
        sys.exit(f"{p}: no channel {name!r}; found {list(d['channels'])}")
    t, v = d["t"], d["channels"][name]
    pv, sd, info = plateau(t, v)
    traces.append((t, v))
    print(f"{p} [{d['format']}, {name}]: plateau {pv*1e3:.4f} mV (within-trace SD {sd*1e3:.4f} mV), "
          f"phase {info['phase_width_s']*1e6:.0f} us, baseline {info['baseline_V']*1e3:.3f} mV")
lsb = lsb_from_settings(vdiv, bits, divs) if vdiv else None
r = plateau_reading(traces, lsb=lsb)
print()
if "too few levels" in r["lsb_from"]:
    print("WARNING: the traces use fewer than 6 ADC levels, so the ADC step cannot be read from the data. "
          "Rerun with --vdiv <volts per division> or the noise floor below is not valid.")
if r["lsb_V"]:
    print(f"ADC step {r['lsb_V']*1e6:.1f} uV ({'from the data' if r['values_on_adc_grid'] and not vdiv else 'from --vdiv'}); "
          f"single-sample noise {r['single_sample_noise_V']*1e6:.1f} uV = {r['noise_over_lsb']:.2f} steps; "
          f"{r['levels_used']} levels used")
    if r["noise_over_lsb"] < 0.5:
        print("  NOTE: noise below half a step -- averaging cannot remove the rounding error "
              f"({r['quantization_V']*1e6:.2f} uV stays). Use a smaller V/div if there is one.")
else:
    print("samples are not on an ADC grid (the scope averaged or used high-resolution mode); "
          "pass --vdiv to include the quantization term")
if r["clipped_fraction"] > 0.01:
    print(f"  WARNING: {r['clipped_fraction']*100:.1f}% of samples sit on the top/bottom level: CLIPPED? "
          "Use a larger V/div and repeat.")
print(f"random part (standard error) {r['standard_error_V']*1e6:.2f} uV; quantization part "
      f"{r['quantization_V']*1e6:.2f} uV")
print(f"\nn_traces {r['n_traces']}  plateau_V {r['plateau_V']:.6g}  plateau_V_sd {r['plateau_V_sd']:.3g}  "
      f"noise_floor_V {r['noise_floor_V']:.3g}")
