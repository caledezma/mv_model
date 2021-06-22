# Minimal model for human ventricular action potentials in tissue

Python implementation of the model for ventricular cells proposed by Bueno-Orovio et. al:

Bueno-Orovio, A., Cherry, E. M., & Fenton, F. H. (2008). Minimal model for human ventricular action potentials in tissue. Journal of theoretical biology, 253(3), 544-560.

The model can be used to reproduce ventricular action potentials of the three different ventricular
cell types (epi, endo, m) or to reproduce other models. For the time being, only single cell
equations have been implemented. The diffusion equation will be implemented in future versions.

# Installing the repo

You can install the `mv_model` package and use the model for your own application by navigating
into the outermost folder and doing:

```
pip install -e .
```

It is suggested that you do this install within a virtual environment. Note that the repo requires
Python 3.8 or greater and there is a requirement on a pre-release version of numpy (1.21.0rc2) due
to the type hints used in the code.

# Example of use

The model is written in a way that is compatible with `scipy.integrate.solve_ivp` solvers. Namely,
the `mv_model.model.mv_model()` function defines the system of ordinary differential equations
that govern the cell's behaviour. SciPy's solver will then solve these equations in time and
provide the result.

You can observe how this is done by runnig the `run_model_example.py` command. To run the examples
you will need to install the repo with the additional packages required for them:

```
pip install -e .[examples]
```

You can then use the CLI to run the model:

```
python examples/run_model_example.py --help
```

and inspect the code if you wish to use a similar solver in your own application. Example results
for running the commands:

```
python examples/run_model_example.py 10 1000 epi examples/figs
python examples/run_model_example.py 10 1000 endo examples/figs
python examples/run_model_example.py 10 1000 m examples/figs
```

is provided in `examples/figs`

A jupyter notebook is also provided. This notebook can be used for a closer inspection of the
outputs of the model.

# Type checking

The repository has been set up for type checking. You can perform your own type checks by doing:

```
pip install -e .[mypy]
mypy . --ignore-missing-imports
```

The missing imports must be ignored because some libraries (matplotlib, scipy, tqdm) don't have
type stubs, so they will produce an error when doing type checks. However, if you run the type
checks without the flag, you should **only** see errors related to missing type hints or library
stubs.
