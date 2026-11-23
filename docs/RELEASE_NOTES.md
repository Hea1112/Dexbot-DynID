# Dexbot-DynID v0.3.1

- Python implementation of closed-chain dynamic parameter identification: QR, WLS, optional LMI and validation.
- Inputs: six-axis kinematics, a matching projected inertial regressor, and measured actuator torques when fitting.
- Drake optional for external robot-model regressor generation; no dependency in the identification pipeline.
- Documentation and automated tests included.

- Release packaging: one unique file per path, data documentation renamed to `data/INPUT_FORMAT.md`.
