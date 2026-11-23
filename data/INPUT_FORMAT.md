# Input format

**Trajectory (CSV):** Time and six-actuator position, velocity and acceleration, in this column order:

```text
time,q0,q1,q2,q3,q4,q5,v0,v1,v2,v3,v4,v5,vd0,vd1,vd2,vd3,vd4,vd5
```

Units: s, rad, rad/s, rad/s². Coordinates: hip pitch / hip roll / hip yaw / knee / ankle LONG rocker / ankle SHORT rocker. A 15 s period or longer is accepted. Optional torque columns: `tau0..tau5`.

**Projected inertial regressor (NPZ):** `Y_in` (or `Y`) with shape `[N,6,60]`, matching the trajectory and actuator order. If only selected samples are exported, include `source_indices` (zero-based original row numbers). Recommended: `time_s`, `paper_q6`, `paper_v6`, `paper_a6`, `parameter_names`. Optional: `theta_nominal`.

**Torque measurements (for WLS/LMI):** `tau0..tau5` in the trajectory or a separate CSV with `time,tau0,tau1,tau2,tau3,tau4,tau5`; timestamps and actuator-side units must agree. Use `--torque-csv` for a separate file. QR does not need measured torques.

**Validation:** An independent time-aligned trajectory, projected regressor and torque measurements.

The inertial regressor must be generated from the robot's model and closed-chain constraint projection outside this package. DynID appends 24 actuator features `[sign(v),v,vd,1]`. For inertia LMI, the 60 inertial columns must be six consecutive 10-parameter link blocks `[m,hx,hy,hz,Ixx,Iyy,Izz,Ixy,Ixz,Iyz]` referenced to each link-frame origin.
