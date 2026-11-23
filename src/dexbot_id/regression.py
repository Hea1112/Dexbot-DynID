"""Paper Eqs. (21)-(26),(30): friction, extended model, observation stack."""
import numpy as np
from .data import ACTUATORS


def friction_block(v, a, deadband=0.0):
    """[sign(vj), vj, aj, 1] per six actuator, shape [N,6,24]."""
    v, a = np.asarray(v, float), np.asarray(a, float)
    if v.ndim != 2 or v.shape[1] != 6 or a.shape != v.shape or deadband < 0:
        raise ValueError("v and a must be [N,6]; deadband >=0")
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("Nonfinite friction input")
    n = len(v)
    phi = np.zeros((n, 6, 24))
    for j in range(6):
        s = np.sign(v[:, j]); s[np.abs(v[:, j]) <= deadband] = 0.0
        phi[:, j, 4*j:4*j+4] = np.stack((s, v[:, j], a[:, j], np.ones(n)), axis=1)
    return phi


def augmented_regressor(Y_in, v, a, deadband=0.0):
    """Y_in must ALREADY be projected into six actuator coordinates.

    For an exact paper-shaped 84-column model, Y_in must have 60 columns.
    No silent 100->60 truncation is performed.
    """
    Y_in = np.asarray(Y_in, float)
    if Y_in.ndim != 3 or Y_in.shape[1] != 6 or Y_in.shape[0] != len(v):
        raise ValueError("Y_in must have shape [N,6,p]")
    if not np.isfinite(Y_in).all():
        raise ValueError("Nonfinite regressor")
    return np.concatenate((Y_in, friction_block(v, a, deadband)), axis=2)


def parameter_names(inertial_names):
    names = [str(x) for x in inertial_names]
    return names + [f"{joint}:{tag}" for joint in ACTUATORS for tag in ("Fc", "Fv", "Ia", "bias")]


def stack(Y, tau):
    Y, tau = np.asarray(Y, float), np.asarray(tau, float)
    if Y.ndim != 3 or tau.shape != Y.shape[:2] or Y.shape[1] != 6:
        raise ValueError("Y [N,6,p] and tau [N,6] required")
    return Y.reshape(-1, Y.shape[-1]), tau.reshape(-1)
