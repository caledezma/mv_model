"""Runs the MV model.
"""
import matplotlib.pyplot as plt
from numpy import array, concatenate
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pdb

from mv_model.model import mv_model
from mv_model.utils import MVParams, get_initial_conditions, get_model_parameters, transform_u_to_ap

CELL_TYPE="m"


params = get_model_parameters(cell_type=CELL_TYPE)
def run_model(num_cycles, cl):
    t = []
    state_vars = []
    currents = []
    y0 = get_initial_conditions()
    for cycle_num in tqdm(range(num_cycles), desc="Computing AP"):
        this_cycle = solve_ivp(
            fun=mv_model,
            t_span=(0, cl),
            y0=y0,
            args=(params, True),
            first_step=0.01,
            max_step=1,
        )
        t.append(array(this_cycle.t) + cl*cycle_num)
        state_vars.append(array(this_cycle.y))
        this_currents = mv_model(
            t=this_cycle.t,
            state_vars=this_cycle.y,
            params=params,
            ret_ode=False
        )
        y0 = [state_var[-1] for state_var in this_cycle.y]
        currents.append(this_currents)
    return concatenate(t), concatenate(state_vars, axis=1).T, concatenate(currents, axis=0)

t1000, state_vars1000, currents1000 = run_model(num_cycles=5, cl=1000)
t500, state_vars500, currents500 = run_model(num_cycles=10, cl=500)

fig = plt.figure()
plt.plot(t1000, transform_u_to_ap(state_vars1000[:,0]))
plt.plot(t500, transform_u_to_ap(state_vars500[:,0]))
plt.savefig(f"ap_{CELL_TYPE}.png")

fig = plt.figure()
plt.plot(t1000, state_vars1000, label=['u', 'v', 'w', 's'])
plt.plot(t500, state_vars500)
plt.savefig(f"state_vars_{CELL_TYPE}.png")
