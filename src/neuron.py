"""Aim 3: does a stimulation pulse make an axon fire?

A straight, unmyelinated axon is split into short compartments. Each has a
Hodgkin-Huxley membrane (sodium, potassium, leak) and is connected to its
neighbours through the axon's internal resistance. The stimulating field
enters through the extracellular potential Ve along the axon:

    C dV_i/dt = - I_ion(V_i) + k (V_{i-1} - 2 V_i + V_{i+1})
                             + k (Ve_{i-1} - 2 Ve_i + Ve_{i+1})

    V   = membrane potential (inside minus outside), mV
    k   = d / (4 R_a dx^2), the axial coupling per unit membrane area
    the last term is the ACTIVATING FUNCTION (Rattay 1986): where the second
    difference of Ve along the axon is positive the membrane is pushed
    toward firing.

Ends are sealed (no current leaves through the cut ends).

Units are the classic Hodgkin-Huxley ones: mV, ms, uA/cm^2, mS/cm^2,
uF/cm^2, cm.

Membranes (choose one):
  HHMembrane       Hodgkin & Huxley 1952 squid axon, 6.3 C. The classic
                   reference. Fails ("heat block") above about 28 C, so it
                   cannot represent mammalian neurons at 37 C.
  MainenMembrane   Mainen & Sejnowski 1996 axon initial segment at 37 C
                   (ModelDB 2488: na.mod, kv.mod, demofig1.hoc): Na 30000 and
                   Kv 2000 pS/um^2, R_a 150 Ohm cm, C_m 0.75 uF/cm^2,
                   R_m 30000 Ohm cm^2, E_Na 60, E_K -90, leak at -70 mV,
                   Na rates shifted -5 mV. The initial segment is the most
                   excitable part of a neuron, so a whole axon with these
                   densities is an upper bound on excitability.
  PassiveMembrane  leak only, for checking the field coupling exactly.

Time stepping (operator splitting, stable for any channel density):
  1. gating variables: exponential-Euler update at the present voltage
  2. ion currents: implicit, each compartment on its own
  3. axial currents + field: implicit, one sparse factorisation reused for
     every axon and every step
Many axons are simulated at once (one column each).

Validated in tests/test_neuron.py: resting state, single-compartment action
potentials against an independent ODE solver for both active membranes,
conduction velocity of the squid giant axon (Hodgkin & Huxley 1952
calculated 18.8 m/s), and the passive cable response to an extracellular
field against the exact cable-equation solution.
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

HH = {"C": 1.0, "gNa": 120.0, "gK": 36.0, "gL": 0.3,
      "ENa": 50.0, "EK": -77.0, "EL": -54.387, "V_rest": -65.0}


def _rates(V):
    """Hodgkin-Huxley opening/closing rates (1/ms) at 6.3 C."""
    with np.errstate(divide="ignore", invalid="ignore"):
        an = np.where(np.abs(V + 55) < 1e-7, 0.1, 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10)))
        am = np.where(np.abs(V + 40) < 1e-7, 1.0, 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10)))
    bn = 0.125 * np.exp(-(V + 65) / 80)
    bm = 4.0 * np.exp(-(V + 65) / 18)
    ah = 0.07 * np.exp(-(V + 65) / 20)
    bh = 1 / (1 + np.exp(-(V + 35) / 10))
    return an, bn, am, bm, ah, bh


def resting_gates(V=HH["V_rest"]):
    an, bn, am, bm, ah, bh = _rates(np.asarray(V, float))
    return an / (an + bn), am / (am + bm), ah / (ah + bh)


def _efun(z):
    """z / (exp(z) - 1), finite at z = 0 (as in the ModelDB files)."""
    z = np.asarray(z, float)
    small = np.abs(z) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = z / np.expm1(z)
    return np.where(small, 1 - z / 2, out)


class HHMembrane:
    """Hodgkin & Huxley 1952 squid axon (rest -65 mV)."""
    name = "Hodgkin-Huxley squid axon"

    def __init__(self, temp_C=6.3):
        self.C = HH["C"]
        self.phi = 3.0 ** ((temp_C - 6.3) / 10)
        self.temp_C = temp_C
        self.V_rest = HH["V_rest"]

    def init_gates(self, V):
        n, m, h = resting_gates(V)
        return [n, m, h]

    def update_gates(self, V, g, dt):
        an, bn, am, bm, ah, bh = _rates(V)
        for x, a, b in zip(g, (an, am, ah), (bn, bm, bh)):
            tau = 1 / (self.phi * (a + b))
            xinf = a / (a + b)
            x[:] = xinf + (x - xinf) * np.exp(-dt / tau)

    def conductances(self, g):
        n, m, h = g
        gna, gk, gl = HH["gNa"] * m ** 3 * h, HH["gK"] * n ** 4, HH["gL"]
        return gna + gk + gl, gna * HH["ENa"] + gk * HH["EK"] + gl * HH["EL"]


class MainenMembrane:
    """Mainen & Sejnowski 1996 axon initial segment, 37 C (ModelDB 2488)."""
    name = "Mainen-Sejnowski axon initial segment, 37 C"
    # na.mod
    tha, qa, Ra, Rb = -35.0, 9.0, 0.182, 0.124
    thi1, thi2, qi, thinf, qinf, Rg, Rd = -50.0, -75.0, 5.0, -65.0, 6.2, 0.0091, 0.024
    # kv.mod
    k_tha, k_qa, k_Ra, k_Rb = 25.0, 9.0, 0.02, 0.002

    def __init__(self, gna_pS_um2=30000.0, gkv_pS_um2=2000.0, celsius=37.0, vshift_na=-5.0):
        self.C = 0.75
        self.tadj = 2.3 ** ((celsius - 23.0) / 10)
        self.gna = gna_pS_um2 * 0.1 * self.tadj            # pS/um^2 -> mS/cm^2, x tadj
        self.gkv = gkv_pS_um2 * 0.1 * self.tadj
        self.gl, self.El = 1e3 / 30000.0, -70.0            # R_m 30000 Ohm cm^2
        self.ENa, self.EK = 60.0, -90.0
        self.vshift = vshift_na
        self.V_rest = self._rest()

    def _na(self, V):
        vm = V + self.vshift
        a = self.Ra * self.qa * _efun((self.tha - vm) / self.qa)
        b = self.Rb * self.qa * _efun((vm - self.tha) / self.qa)
        mtau, minf = 1 / self.tadj / (a + b), a / (a + b)
        a = self.Rd * self.qi * _efun((self.thi1 - vm) / self.qi)
        b = self.Rg * self.qi * _efun((vm - self.thi2) / self.qi)
        htau, hinf = 1 / self.tadj / (a + b), 1 / (1 + np.exp(np.clip((vm - self.thinf) / self.qinf, -700, 700)))
        return minf, mtau, hinf, htau

    def _kv(self, V):
        a = self.k_Ra * self.k_qa * _efun(-(V - self.k_tha) / self.k_qa)
        b = self.k_Rb * self.k_qa * _efun((V - self.k_tha) / self.k_qa)
        return a / (a + b), 1 / self.tadj / (a + b)

    def _I_ss(self, V):
        """Steady-state membrane current (all gates at their resting values)."""
        minf, _, hinf, _ = self._na(np.asarray(V, float))
        ninf, _ = self._kv(np.asarray(V, float))
        return (self.gna * minf ** 3 * hinf * (V - self.ENa)
                + self.gkv * ninf * (V - self.EK) + self.gl * (V - self.El))

    def _rest(self):
        """Stable resting potential: where the steady-state current crosses zero
        going upward (outward current pushes the voltage back down), closest to
        the leak reversal of -70 mV. In the published model the initial segment
        is held near -70 mV by the soma; an axon made entirely of
        initial-segment membrane settles at its own stable rest instead."""
        from scipy.optimize import brentq
        Vs = np.linspace(-100.0, -40.0, 1201)
        I = self._I_ss(Vs)
        idx = np.flatnonzero((I[:-1] < 0) & (I[1:] >= 0))
        if len(idx) == 0:
            raise ValueError("no stable resting potential between -100 and -40 mV")
        roots = [brentq(lambda v: float(self._I_ss(v)), Vs[i], Vs[i + 1]) for i in idx]
        return min(roots, key=lambda v: abs(v - self.El))

    def init_gates(self, V):
        minf, _, hinf, _ = self._na(np.asarray(V, float))
        ninf, _ = self._kv(np.asarray(V, float))
        return [minf, hinf, ninf]

    def update_gates(self, V, g, dt):
        minf, mtau, hinf, htau = self._na(V)
        ninf, ntau = self._kv(V)
        for x, xinf, tau in zip(g, (minf, hinf, ninf), (mtau, htau, ntau)):
            x[:] = xinf + (x - xinf) * np.exp(-dt / tau)

    def conductances(self, g):
        m, h, n = g
        gna, gk = self.gna * m ** 3 * h, self.gkv * n
        return gna + gk + self.gl, gna * self.ENa + gk * self.EK + self.gl * self.El


class SegmentedMembrane(MainenMembrane):
    """Mainen-Sejnowski channels with a sodium density that varies along the
    axon (e.g. an initial segment at 30000 pS/um^2 followed by axon at 8000).
    gna_pS_um2: one value per compartment. The resting potential differs
    between segments, so use Axon.settle() before stimulating."""

    def __init__(self, gna_pS_um2, gkv_pS_um2=2000.0, celsius=37.0, vshift_na=-5.0):
        g = np.asarray(gna_pS_um2, float)
        rests = {v: MainenMembrane(v, gkv_pS_um2, celsius, vshift_na).V_rest for v in np.unique(g)}
        super().__init__(float(g[0]), gkv_pS_um2, celsius, vshift_na)
        self.gna = (g * 0.1 * self.tadj)[:, None]
        self.V_rest = np.array([rests[v] for v in g])[:, None]
        self.name = "Mainen-Sejnowski, segmented sodium density, 37 C"


class PassiveMembrane:
    name = "passive"

    def __init__(self, gL=0.3, EL=-65.0, C=1.0):
        self.C, self.gl, self.El, self.V_rest = C, gL, EL, EL

    def init_gates(self, V):
        return []

    def update_gates(self, V, g, dt):
        pass

    def conductances(self, g):
        return self.gl, self.gl * self.El


class Axon:
    """A straight cable of n compartments, length dx_cm each, diameter d_cm,
    axial resistivity Ra_ohm_cm, with the given membrane (default: HH)."""

    def __init__(self, n, dx_cm, d_cm, Ra_ohm_cm=35.4, dt_ms=0.001, membrane=None,
                 temp_C=6.3):
        self.n, self.dx, self.d, self.Ra, self.dt = n, dx_cm, d_cm, Ra_ohm_cm, dt_ms
        self.mem = membrane if membrane is not None else HHMembrane(temp_C)
        # axial coupling per unit membrane area, mS/cm^2
        self.k = self.d / (4 * self.Ra * self.dx ** 2) * 1e3
        if n == 1:                          # a single isolated compartment
            lap = diags([np.zeros(1)], [0]).tolil()
        else:
            lap = diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1]).tolil()
            lap[0, 0] = -1                  # sealed ends
            lap[n - 1, n - 1] = -1
        self.lap = lap.tocsc()
        M = (self.mem.C / self.dt) * diags([np.ones(n)], [0]) - self.k * self.lap
        self.lu = splu(M.tocsc())

    def run(self, t_end_ms, Ve_fn=None, I_inj_fn=None, record=None, n_axons=1,
            fire_check=None):
        """Integrate from rest.

        Ve_fn(step) -> (n, n_axons) extracellular potential in mV at the END of
            the step (or None for no field)
        I_inj_fn(step) -> (n, n_axons) injected current density, uA/cm^2
        record: list of compartment indices to store V(t) for (or None)
        fire_check: indices of compartments; returns, per axon, whether any of
            them ever exceeded 0 mV (an action potential arrived there)
        """
        n, dt, mem = self.n, self.dt, self.mem
        Cdt = mem.C / dt
        V0 = getattr(self, "V0", None)
        if V0 is None:
            V0 = np.broadcast_to(np.asarray(mem.V_rest, float).reshape(-1, 1) if np.ndim(mem.V_rest)
                                 else mem.V_rest, (n, 1))
            g0 = mem.init_gates(V0)
        else:
            g0 = self.g0
        V = np.array(np.broadcast_to(V0, (n, n_axons)), float)
        gates = [np.array(np.broadcast_to(g, V.shape), float) for g in g0]
        steps = int(round(t_end_ms / dt))
        traces = [] if record is not None else None
        fired = np.zeros(n_axons, dtype=bool)
        for s in range(steps):
            # 1. gates
            mem.update_gates(V, gates, dt)
            # 2. ion currents, implicit per compartment
            G, GE = mem.conductances(gates)
            num = Cdt * V + GE
            if I_inj_fn is not None:
                num = num + I_inj_fn(s)
            V = num / (Cdt + G)
            # 3. axial currents + field, implicit
            rhs = Cdt * V
            if Ve_fn is not None:
                Ve = Ve_fn(s)
                if Ve is not None:
                    rhs = rhs + self.k * (self.lap @ Ve)
            V = self.lu.solve(rhs)
            if V.ndim == 1:
                V = V[:, None]
            if fire_check is not None:
                fired |= np.any(V[fire_check] > 0.0, axis=0)
            if record is not None:
                traces.append(V[record].copy())
        out = {"V": V, "fired": fired}
        if record is not None:
            out["t_ms"] = (np.arange(steps) + 1) * dt
            out["traces"] = np.array(traces)
        return out


def settle(axon, t_ms=20.0):
    """Run one unstimulated axon for t_ms and keep its final state as the
    starting state of every later run (needed when the resting potential
    differs along the axon). Returns the largest |dV/dt| at the end (mV/ms)."""
    axon.V0 = None
    out = axon.run(t_ms - axon.dt, n_axons=1)
    V1 = out["V"]
    # one more step to measure the remaining drift, with the gates carried on
    n, mem = axon.n, axon.mem
    g = mem.init_gates(V1)            # quasi-steady gates at the settled voltage
    axon.V0, axon.g0 = V1, g
    V2 = axon.run(axon.dt * 10, n_axons=1)["V"]
    axon.V0, axon.g0 = V2, mem.init_gates(V2)
    return float(np.max(np.abs(V2 - V1)) / (axon.dt * 10))


def straight_axon_points(center_xyz, direction_xy, length_m, n, start=False):
    """Coordinates (n, 3) of compartment centres along a straight axon lying
    parallel to the chip, centred on center_xyz (or STARTING there, if
    start=True: then compartment 0 is at center_xyz)."""
    ux, uy = direction_xy / np.hypot(*direction_xy)
    s = (np.arange(n) + 0.5) * (length_m / n) if start else (np.arange(n) - (n - 1) / 2) * (length_m / n)
    return np.column_stack([center_xyz[0] + s * ux, center_xyz[1] + s * uy,
                            np.full(n, center_xyz[2])]), s


def thresholds(axon, Phi_V_per_A, i_unit_A, t_end_ms, fire_check, amp_max_V=3.0,
               iterations=12):
    """Lowest stimulus amplitude (V) that makes each axon fire.

    Phi_V_per_A: (n, n_axons) field along each axon per ampere delivered
    i_unit_A:    current per step for a 1 V pulse, one value per time step
    Bisection on the amplitude, all axons at once. Returns NaN where even
    amp_max_V does not fire.
    """
    n_ax = Phi_V_per_A.shape[1]
    Phi_mV = Phi_V_per_A * 1e3
    lo = np.zeros(n_ax)
    hi = np.full(n_ax, amp_max_V)

    def fires(amps):
        def Ve(s):
            if i_unit_A[s] == 0 and (s == 0 or i_unit_A[s - 1] == 0):
                return None
            return Phi_mV * (amps * i_unit_A[s])[None, :]
        return axon.run(t_end_ms, Ve_fn=lambda s: Ve(s) if s < len(i_unit_A) else None,
                        n_axons=n_ax, fire_check=fire_check)["fired"]

    top = fires(hi)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        f = fires(mid)
        hi = np.where(f, mid, hi)
        lo = np.where(f, lo, mid)
    return np.where(top, hi, np.nan)
