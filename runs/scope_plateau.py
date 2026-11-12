"""Plateau potential from oscilloscope trace files (plan steps 53-54).

    python runs/scope_plateau.py data/phantom/raw/s1_pos03_200us_*.csv

Prints, for each file, the first-phase plateau (baseline subtracted, middle
half of the phase) and then the mean and SD over all files given -- paste
those into the measurements CSV (plateau_V, plateau_V_sd, n_traces).
Raw files are never modified.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from scope import read_trace, plateau

files = sys.argv[1:]
if not files:
    sys.exit(__doc__)
vals = []
for p in files:
    t, v = read_trace(p)
    pv, sd, info = plateau(t, v)
    vals.append(pv)
    print(f"{p}: plateau {pv*1e3:.4f} mV (within-trace SD {sd*1e3:.4f} mV), phase "
          f"{info['phase_width_s']*1e6:.0f} us, baseline {info['baseline_V']*1e3:.3f} mV")
vals = np.array(vals)
print(f"\nn_traces {len(vals)}  plateau_V {vals.mean():.6g}  plateau_V_sd "
      f"{vals.std(ddof=1) if len(vals) > 1 else 0:.3g}")
