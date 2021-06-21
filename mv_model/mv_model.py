"""Defines the ODEs for the minimal ventricular model.
"""
from typing import NamedTuple, List

class MVParams(NamedTuple):
    """Parameters for the MV model, defaults to endocardial cells.
    """
    u_o: float = 0
    u_u: float = 1.55
    th_v: float = 0.3
    th_w: float = 0.13
    th_v_minus: float = 0.006
    th_o: float = 0.006
    tau_v1: float = 60
    tau_v2: float = 1150
    tau_v: float = 1.4506
    tau_w1: float = 60
    tau_w2: float = 15
    kappa_w: float = 65
    u_w: float = 0.03
    tau_w: float = 200
    tau_fi: float = 0.11
    tau_o1: float = 400
    tau_o2: float = 6
    tau_so1: float = 30.0181
    tau_so2: float = 0.9957
    kappa_so: float = 2.0458
    u_so: float = 0.65
    tau_s1: float = 2.7342
    tau_s2: float = 16
    kappa_s: float = 2.0994
    u_s: float = 0.9087
    tau_si: float = 1.8875
    tau_w_inf: float = 0.07
    w_inf_star: float = 0.94


def heaviside(x):
    """Standard heaviside function.
    """
    return 1 if x >= 1 else 0

def mv_model(
    t: float,
    state_vars: List[float],
    params: MVParams,
):
    """ODEs for the minimal ventricular model.
    """
    u, v, w, s = state_vars

    tv_minus = (
        (1-heaviside(u-params.thv_minus)) * params.tauv1_minus +
        heaviside(u - params)
    )
