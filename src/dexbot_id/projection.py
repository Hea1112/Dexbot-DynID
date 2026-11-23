"""Closed-chain virtual-work projection, manuscript Eqs. (9)-(20)."""
import numpy as np


def tangent_map(Ja, Ju, jdot_qdot=None):
    """For q=[q_a;q_u], J=[Ja,Ju] with square, regular Ju.

    Returns G=[I;Gamma], where Gamma=-solve(Ju,Ja). The acceleration bias
    gdot_qdot=[0;-solve(Ju,Jdot*qdot)] can be formed when Jdot*qdot is known.
    """
    Ja, Ju = np.asarray(Ja, float), np.asarray(Ju, float)
    if Ja.ndim != 2 or Ju.ndim != 2 or Ju.shape[0] != Ju.shape[1] or Ja.shape[0] != Ju.shape[0]:
        raise ValueError("Expected Ja [m,na], Ju square [m,m]")
    if np.linalg.cond(Ju) > 1e12:
        raise ValueError("Passive Jacobian near-singular")
    gamma = -np.linalg.solve(Ju, Ja)
    G = np.vstack((np.eye(Ja.shape[1]), gamma))
    bias = np.zeros(G.shape[0])
    if jdot_qdot is not None:
        w = np.asarray(jdot_qdot, float)
        if w.shape != (Ju.shape[0],):
            raise ValueError("Jdot*qdot shape mismatch")
        bias[Ja.shape[1]:] = -np.linalg.solve(Ju, w)
    return G, bias


def full_kinematics(G, bias, qa_dot, qa_ddot):
    G, bias = np.asarray(G, float), np.asarray(bias, float)
    v, a = G @ np.asarray(qa_dot, float), G @ np.asarray(qa_ddot, float) + bias
    return v, a


def project_full_regressor(Y_full, G):
    """GᵀY_full eliminates ideal constraint reactions (paper Eq. 17–25)."""
    Y, G = np.asarray(Y_full, float), np.asarray(G, float)
    if Y.ndim != 2 or Y.shape[0] != G.shape[0]:
        raise ValueError("Full regressor generalized rows must match G")
    return G.T @ Y


def projected_torque(full_generalized_force, G):
    return np.asarray(G, float).T @ np.asarray(full_generalized_force, float)
