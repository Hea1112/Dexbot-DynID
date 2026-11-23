# Manuscript equation and implementation map

The implementation is independently authored and provides the numerical identification-method components, not the original unpublished experimental code. No demonstration inputs or output datasets are distributed.

| Manuscript topic | Implementation | Input and limitation |
|---|---|---|
| 2-RSS+1U ankle, constraint equations | `dexbot_id/projection.py` | Generic Jacobian/tangent algebra; **actual 2-RSS+1U geometry must be provided by an external robot model** |
| Eqs. (9)–(14) | `projection.tangent_map`, `full_kinematics` | `Ja`, `Ju`, `Jdot*qdot` curvature term and independent q/v/a |
| Eqs. (15)–(20) | `projection.project_full_regressor` | Apply `G.T` to physically derived full model regressor |
| Drive model Eqs. (21)–(22) | `regression.friction_block` | Four columns per actuator: Coulomb, viscous, rotor acceleration, bias |
| Combined regression Eqs. (23)–(25) | `regression.augmented_regressor` | Requires **externally supplied** 6×60 projected inertial regressor to obtain 84 columns |
| Numerical base-parameter selection Eq. (26) | `base.select_base_columns` | Pivoted QR and SVD; rank is computed, **not** fixed to manuscript-reported 59 |
| Fourier trajectory Eqs. (27)–(28) | `fourier.evaluate`; `excitation_optimization.optimize_fourier` | Robot-specific objective/limits and approved reference coefficients not included |
| Acceleration filtering Eq. (29) | `preprocessing` | 4th-order differentiation, optional zero-phase filtering; changes to states require recomputing the regressor |
| Stacked observations Eq. (30) | `regression.stack` | Time-aligned `Y_in` and optional `tau0..tau5` observations |
| Weighted least squares Eq. (31) | `wls.solve_wls` | Channel-RMS weights and stable solve |
| Physical consistency Eqs. (32)–(34) | `physical.solve_lmi_wls` | Optional CVXPY SDP on **full** per-link inertia tensor, not directly on base coefficients |
| Independent validation section | `validation.rmse_by_actuator` | A separate trajectory CSV **plus** its projected regressor NPZ |

The manuscript's 60-inertia/24-drive/84-extended/59-base values represent the publication's model and computation, not values manufactured by this package. The exact 84→59 reduction requires publication-specific model/regressor data and numerical tolerance.

## Parameter conventions for SDP

One physical moving link contributes `[m, hx, hy, hz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`, with inertia tensor about the **link origin**. Let its origin inertia be `L` and first moment `h`. The affine pseudo-inertia is `[[trace(L)/2 I-L, h], [h.T,m]]`. A 4×4 semidefinite constraint and nonnegative mass are enforced on the **full** inertia parameters; rotor inertia, Coulomb and viscous coefficients can be constrained nonnegative. Numerical base parameters do not uniquely determine a physical full parameter vector without a feasible lifting.

## CSV indexing and validation

The trajectory CSV adopts hip pitch/roll/yaw, knee, two rocker angles, even though Eq. (1)'s shorthand differs. Its sample interval is nominally 0.002 s. One 15-second period or the complete 60-second acquisition is accepted. Torque columns are optional for trajectory and QR analysis. Only a new recording/trajectory with a matching projected regressor supports independent prediction statistics.
