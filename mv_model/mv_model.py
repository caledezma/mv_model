"""Defines the ODEs for the minimal ventricular model.
"""
from typing import NamedTuple, List

import numpy as np

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

    tau_v_minus = (
        (1-heaviside(u-params.tau_v)) * params.tau_v1 +
        heaviside(u - params.tau_v) * params.tau_v2
    )
    tau_w_minus = (
        params.tau_w1 +
        (params.tau_w2 - params.tau_w1) * (1+np.tanh(params.kappa_w*(u - params.u_w))) / 2
    )
    tau_so = (
        params.tau_so1 +
        (params.tau_so2 - params.tau_so1)*(1+np.tanh(params.kappa_so*(u-params.u_so))) / 2
    )
    tau_s = (
        (1 - heaviside(u-params.th_w))*params.tau_s1 +
        heaviside(u-params.tau_w)*params.tau_s2
    )
    tau_o = (
        (1 - heaviside(u-params.th_o))*params.tau_o1 +
        heaviside(u-params.th_o)*params.tau_o2
    )
    v_inf = 1 if u < params.th_v_minus else 0
    w_inf = (
        (1 - heaviside(u-params.th_o))*(1-u/params.tau_w_inf) +
        heaviside(u-params.th_o)*params.w_inf_star
    )
    Jfi = -v*heaviside(u-params.th_v)*(u-params.th_v)*(params.u_u-u)/params.tau_fi
    Jso = (u-params.u_o)*(1-heaviside(u-params.th_w)) / tau_o - heaviside(u-params.th_w)/tau_so
    Jsi = -heaviside(u-params.th_w)*w*s/params.tau_si

    du = -(Jfi + Jso + Jsi)
    dv = (
        (1-heaviside(u-params.th_v))*(v_inf-v)/tau_v_minus -
        heaviside(u-params.th_v)*v/params.tau_w
    )
    dw = (
        (1-heaviside(u-params.th_w))*(w_inf-w)/tau_w_minus -
        heaviside(u-params.th_w)*w/params.th_w
    )
    ds = ((1+np.tanh(params.kappa_s*(u-params.u_s)))/2 - s)/tau_s

    return [du, dv, dw, ds]