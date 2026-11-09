"""V3: mode consistency.

1. CURRENT mode: drive the disk with I, read its potential V_e.
2. VOLTAGE mode: hold the disk at V_e.
The recovered current must equal I, and the whole field must match, within 1%.
This is the best single check that the fixed-voltage (Dirichlet) electrode
and the floating (current-driven) electrode are the same physics.
Outer faces are grounded at 0 V (a simple return), so nothing is borrowed
from the exact solution.
"""
import numpy as np
from disk_setup import solve_disk

I = 10e-6
rc = solve_disk("current", I, outer="ground", size=15)
V_e = rc["stim_potential_V"]
rv = solve_disk("voltage", V_e, outer="ground", size=15)

err_I = abs(rv["delivered_current_A"] / I - 1)
diff = np.max(np.abs(rv["phi"] - rc["phi"])) / np.max(np.abs(rc["phi"]))

print(f"current mode: I = {I*1e6:.3f} uA  ->  electrode at {V_e*1e3:.4f} mV")
print(f"voltage mode: {V_e*1e3:.4f} mV  ->  delivered {rv['delivered_current_A']*1e6:.4f} uA "
      f"({(rv['delivered_current_A']/I-1)*100:+.4f}%)")
print(f"largest field difference anywhere: {diff*100:.4f}% of peak")
print(f"current check in current mode: delivered {rc['delivered_current_A']*1e6:.4f} uA")
print("PASS" if err_I < 0.01 and diff < 0.01 else "FAIL", "(criterion < 1%)")
