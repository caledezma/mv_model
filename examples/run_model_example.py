"""Runs the MV model.
"""
import os
from typing import Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from tqdm import tqdm

from mv_model.model import mv_model
from mv_model.utils import MVParams, get_initial_conditions, get_model_parameters, transform_u_to_ap


def run_model(
    num_cycles: int,
    cycle_length: int,
    params: MVParams
) -> Tuple[npt.NDArray[np.float_], ...]:
    t = []
    state_vars = []
    currents = []
    y0 = get_initial_conditions()
    for cycle_num in tqdm(range(num_cycles), desc="Computing AP signal"):
        this_cycle = solve_ivp(
            fun=mv_model,
            t_span=(0, cycle_length),
            y0=y0,
            args=(params, True),
            first_step=0.01,
            max_step=1,
        )
        t.append(np.array(this_cycle.t) + cycle_length*cycle_num)
        state_vars.append(np.array(this_cycle.y))
        this_currents = mv_model(
            t=this_cycle.t,
            state_vars=this_cycle.y,
            params=params,
            ret_ode=False
        )
        y0 = [state_var[-1] for state_var in this_cycle.y]
        currents.append(this_currents)
    return np.concatenate(t), np.concatenate(state_vars, axis=1).T, np.concatenate(currents, axis=0)


@click.command()
@click.argument("num_cycles", type=int)
@click.argument("cycle_length", type=int)
@click.argument("cell_type", type=click.Choice(['epi', 'endo', 'm'], case_sensitive=True))
@click.argument("outdir", type=click.Path(exists=True))
def test_mv_model(num_cycles, cycle_length, cell_type, outdir):
    """Run the minimal model for human ventricular action potentials in tissue with a regular
    stimulation pattern. Use this CLI to test the different cell types at different cycle lengths
    and for different numbers of cycles.

    NUM_CYCLES number of cycles to simulate.
    CYCLE_LENGTH time, in ms, between stimulations.
    CELL_TYPE which myocardial cell is being simulated.
    OUTDIR directory where the outputs of the model will be saved.
    """
    t, state_vars, currents = run_model(
        num_cycles=num_cycles,
        cycle_length=cycle_length,
        params=get_model_parameters(cell_type=cell_type)
    )
    file_end = f"{cell_type}_{cycle_length}cl_{num_cycles}cycles.png"
    plt.figure(figsize=(20, 5))
    plt.plot(t, transform_u_to_ap(state_vars[:,0]))
    plt.title("Action potential produced by the Minimal Model")
    plt.ylabel("Action Potential (mV)")
    plt.xlabel("Time (ms)")
    plt.savefig(os.path.join(outdir, f"ap_{file_end}"))

    plt.figure(figsize=(20, 5))
    plt.plot(t, state_vars[:,1:], label=['v', 'w', 's'])
    plt.title("State variables of the Minimal Model")
    plt.ylabel("Value (dimensionless)")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"state_vars_{file_end}"))

    plt.figure(figsize=(20, 5))
    plt.plot(t, currents, label=['J_{fi}', 'J_{so}', 'J_{si}', 'I_{stim}'])
    plt.title("Currents of the Minimal Model")
    plt.ylabel("Value (dimensionless)")
    plt.xlabel("Time (ms)")
    plt.legend()
    plt.savefig(os.path.join(outdir, f"currents_{file_end}"))


if __name__ == "__main__":
    test_mv_model()
