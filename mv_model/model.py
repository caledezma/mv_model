"""Defines the ODEs for the minimal ventricular model.
"""
from typing import NamedTuple, List

import numpy as np

from .utils import MVParams, heaviside

def mv_model(
    t: float,
    state_vars: List[float],
    params: MVParams,
    ret_ode: bool,
):
    """ODEs for the minimal ventricular model.
    """
    u, v, w, s = state_vars

    tau_v_minus = (
        (1-heaviside(u-params.th_v_minus)) * params.tau_v1 +
        heaviside(u - params.th_v_minus) * params.tau_v2
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
        heaviside(u-params.th_w)*params.tau_s2
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
    Jso = (u-params.u_o)*(1-heaviside(u-params.th_w)) / tau_o + heaviside(u-params.th_w)/tau_so
    Jsi = -heaviside(u-params.th_w)*w*s/params.tau_si

    Jstim = 0.4 if t < 1 else 0

    du = -(Jfi + Jso + Jsi) + Jstim
    dv = (
        (1-heaviside(u-params.th_v)) * (v_inf-v)/tau_v_minus -
        heaviside(u-params.th_v)*v/params.tau_v
    )
    dw = (
        (1-heaviside(u-params.th_w)) * (w_inf-w)/tau_w_minus -
        heaviside(u-params.th_w)*w/params.tau_w
    )
    ds = ((1+np.tanh(params.kappa_s*(u-params.u_s)))/2 - s)/tau_s

    if ret_ode:
        return [du, dv, dw, ds]
    return [Jfi, Jso, Jsi, Jstim]