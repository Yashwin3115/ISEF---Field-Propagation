"""Check that the solver refuses bad inputs with a clear message,
instead of running and returning a wrong answer."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fd3d import solve_3d

x = y = z = np.linspace(0, 1, 6)
shape = (6, 6, 6)
good = dict(sigma=np.ones(shape), source=np.zeros(shape), phi_fixed=np.zeros(shape))


def expect_error(label, xx=x, fixed=None, **changes):
    args = {**good, **changes}
    try:
        solve_3d(xx, y, z, args["sigma"], args["source"], args["phi_fixed"], fixed)
    except ValueError as e:
        print(f"  caught  {label:<32} -> {e}")
        return True
    print(f"  MISSED  {label}")
    return False


bad_sigma = np.ones(shape); bad_sigma[2, 2, 2] = 0.0
nan_src = np.zeros(shape); nan_src[3, 3, 3] = np.nan
inf_phi = np.zeros(shape); inf_phi[0, 0, 0] = np.inf

checks = [
    expect_error("zero conductivity", sigma=bad_sigma),
    expect_error("negative conductivity", sigma=-np.ones(shape)),
    expect_error("NaN in source", source=nan_src),
    expect_error("infinite boundary value", phi_fixed=inf_phi),
    expect_error("wrong array shape", sigma=np.ones((5, 6, 6))),
    expect_error("repeated grid node", xx=np.array([0, .2, .2, .6, .8, 1.])),
    expect_error("no fixed nodes", fixed=np.zeros(shape, dtype=bool)),
]

# and a good input must still solve
phi, st = solve_3d(x, y, z, **good)
ok_good = np.allclose(phi, 0)
print(f"  good input solves: {ok_good}")

print("PASS" if all(checks) and ok_good else "FAIL")
