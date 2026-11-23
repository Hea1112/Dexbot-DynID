# Model-specific inertia regressor interface

The public method framework deliberately does **not** include unpublished robot or hardware dependencies. Its responsibility begins after a physically correct model-specific regression matrix has been generated.

## Required computation outside this package

For each sample, a physical dynamics exporter must compute a constrained full-link inertial regressor in the chosen joint coordinates, then project through the closed ankle's constraint-consistent tangent mapping (virtual work). Its output is the **six actuator-coordinate inertial regressor** `Y_in(t)`.

The 6 output rows must correspond, in order, to hip pitch, hip roll, hip yaw, knee, left ankle LONG rocker and SHORT rocker. The number and ordering of inertial columns must match the actual manuscript model. For the stated 84-column extension, **60 inertial columns** are needed; this repository adds the 24 friction/rotor/bias columns.

In schematic form, if `q_full = f(q_drive)` and `dq_full = G dq_drive`, the projected inertial columns are `Y_drive = G.T @ Y_full` for a correctly assembled full regressor (with the appropriate constrained accelerations). Reaction forces from ideal constraints then do no work in the independent coordinates.

## Required NPZ

- `Y_in` or `Y`: `[N,6,60]` float array (or another column count if not claiming 84 columns).
- `source_indices` if sampling from a longer CSV.
- Recommended: `time_s`, `paper_q6`, `paper_v6`, `paper_a6`, `parameter_names`, `theta_nominal`.

## Can Drake be used?

Yes, **outside this core package**. The exporter could compute multibody state, spatial kinematics and link inertial basis forces using Drake or the original robot project. An independently implemented Newton–Euler or analytical approach is also possible. The repository does not import Drake, does not auto-run model-specific C++ binaries, and does not bundle unpublished robot URDFs.

**Neither a robot URDF alone nor a CSV alone is a substitute for the matching projected inertial regressor.** An externally generated matrix must be checked for consistent units, joint order, frame conventions, and mass/inertia definitions before interpretation as the paper's own model.
