# Dexbot-DynID

**Dynamic identification of the Dexbot lower limb with closed-chain ankle constraints.**

## Method

Fourier excitation · Closed-chain projection · Inertial and friction regression · QR base-parameter reduction · WLS · Optional LMI · Torque validation.

## Install

Python 3.8+, NumPy and SciPy are required. CVXPY is optional for LMI optimization.

```bash
python3 -m pip install -e .
python3 -m pytest -q
```

## Run

```bash
python3 scripts/run_identification.py --csv trajectory.csv --inertia-npz regressor.npz --require-84 --output results
```

Add `--torque-csv torques.csv` for parameter estimation.

**Drake:** Optional for model-based construction of the projected inertial regressor. The identification package itself does not require Drake or the robot control software.

[Method](docs/METHOD_TO_CODE.md) · [Interface](docs/REGRESSOR_INTERFACE.md)
