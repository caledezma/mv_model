"""Runs the MV model.
"""
import os

import click
import matplotlib.pyplot as plt

from mv_model.utils import transform_u_to_ap
from examples.utils import run_model


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
        cell_type=cell_type
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
