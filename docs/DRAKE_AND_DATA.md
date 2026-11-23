# Drake integration

**Drake is optional.** The identification package uses Python, NumPy and SciPy; CVXPY plus an SDP solver is only needed for LMI.

For a robot model, Drake can provide multibody kinematics and dynamics. A separate, robot-specific exporter must account for the 2-RSS+1U closed ankle, project inertial generalized forces onto the six independent drives, and write the time-aligned inertial regressor (`Y_in`, shape `[N,6,60]` for the paper's column convention).

DynID reads this exported matrix and the joint trajectory; it does not invoke Drake, load robot URDFs or build the controller software. Torque measurements are only required for WLS/LMI fitting and quantitative validation.
