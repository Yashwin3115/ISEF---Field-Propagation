"""Measure peak memory of one 3D build + CG/AMG solve. Run one size per process:
    python tests/measure_memory.py 100
"""
import sys, resource, platform
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import solve_3d

N = int(sys.argv[1])
x = y = z = np.linspace(0, 1, N + 1)
shape = (N + 1,) * 3
rng = np.random.default_rng(0)
sigma = 1 + rng.random(shape)
phi_b = np.zeros(shape)
src = np.zeros(shape); src[N // 2, N // 2, N // 2] = 1.0
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
phi, st = solve_3d(x, y, z, sigma, src, phi_b)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
scale = 1 if platform.system() == "Darwin" else 1024     # macOS reports bytes, Linux KB
m = st["unknowns"]
print(f"N={N:4d} unknowns={m:>10,} peak={peak*scale/1e9:6.2f} GB "
      f"bytes/unknown={(peak - before)*scale/m:6.0f} iters={st['iterations']}")