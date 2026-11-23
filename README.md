# Dexbot-DynID

**Dynamic Parameter Identification for the Dexbot Lower Limb**

Python tools for the method described in *Identification of lower-limb dynamic parameters in a humanoid robot based on closed-loop ankle constraint projections*.

## Method

Fourier excitation · Closed-chain constraint projection · Inertial and drive-friction regression · QR base-parameter selection · Weighted least squares (WLS) · Optional physical-consistency LMI · Torque prediction and validation.

## Requirements

- **Python 3.8+**, NumPy, SciPy
- **CVXPY and an SDP solver** (optional, for LMI)
- **Drake** (optional, for model-based regressor generation; **not required by the identification package**)

Drake can calculate robot kinematics and inertial dynamics from a URDF/multibody model. A model-specific exporter must account for the closed-chain ankle constraints and produce a regressor in the six independent actuator coordinates. The exporter is separate from this package; no Drake or robot-control project is needed to run QR/WLS/LMI on the exported data.

## Input data

1. **Joint trajectories:** time, positions, velocities and accelerations of six actuators; one 15 s period or longer.
2. **Projected inertia regressor:** time-aligned array `[N, 6, 60]` for the manuscript's 60-parameter inertial model. DynID adds 24 drive terms to form 84 columns.
3. **Actuator torques:** needed for WLS/LMI fitting; not required for QR analysis. Independent validation uses a separate trajectory, regressor and torque observations.

Joint order: hip pitch, hip roll, hip yaw, knee, left ankle LONG rocker, left ankle SHORT rocker. See [input format](data/INPUT_FORMAT.md) and [regressor interface](docs/REGRESSOR_INTERFACE.md).

## Install and run

```bash
python3 -m pip install -e .
python3 scripts/check_data.py --csv trajectory.csv
python3 scripts/run_identification.py --csv trajectory.csv --inertia-npz inertia_regressor.npz --require-84 --output results
```

Provide `--torque-csv torques.csv` for WLS; add `--lmi --n-links 6` after `pip install -e '.[sdp]'` for physical-consistency optimization. Run tests with `pip install -e '.[test]'` and `python3 -m pytest -q`.

The numerical base rank is computed from the supplied regressor; the manuscript's reported 59 parameters are not hard-coded.

**Documentation:** [Method map](docs/METHOD_TO_CODE.md) · [Drake interface](docs/DRAKE_AND_DATA.md) · [GitHub publication](docs/GITHUB_PUBLISH.md)
