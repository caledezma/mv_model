"""Runs the MV model.
"""
import matplotlib.pyplot as plt
from numpy import array, concatenate
from scipy.integrate import solve_ivp
from tqdm import tqdm

from mv_model.model import mv_model
from mv_model.utils import MVParams, get_initial_conditions, get_model_parameters

CELL_TYPE="m"

params = get_model_parameters(cell_type=CELL_TYPE)
def run_model(num_cycles, cl):
    t = []
    ap = []
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
        ap.append(array(this_cycle.y[0])*85.7 - 84)
        y0 = [state_var[-1] for state_var in this_cycle.y]
    return concatenate(t), concatenate(ap)

t1000, ap1000 = run_model(num_cycles=5, cl=1000)
t500, ap500 = run_model(num_cycles=10, cl=500)

fig = plt.figure()
plt.plot(t1000, ap1000)
plt.plot(t500, ap500)
plt.savefig(f"ap_{CELL_TYPE}.png")
