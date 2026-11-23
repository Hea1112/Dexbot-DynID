# Dependencies

- **Required:** Python 3.8+, NumPy, SciPy.
- **Optional:** CVXPY and SDP solver for LMI (`pip install -e '.[sdp]'`).
- **Tests:** pytest (`pip install -e '.[test]'`).
- **Drake:** optional external tool for generating the robot-specific projected inertial regressor. DynID does not import Drake or depend on the robot control code.
