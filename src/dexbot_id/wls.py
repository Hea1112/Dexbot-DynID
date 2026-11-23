"""Paper Eq. (31): channel-balanced WLS with stable square-root least squares."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lstsq


@dataclass
class Fit:
    coefficients: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    channel_weights: np.ndarray
    weighted_rmse: float
    rank: int


def channel_weights(tau):
    """w_j=1/(s_j^2 + median(s_j)^2), per manuscript section 4.5.

    This is defined for physical torque units used consistently per channel.
    """
    y = np.asarray(tau, float)
    if y.ndim != 2 or y.shape[1] != 6 or not np.isfinite(y).all():
        raise ValueError("Need finite measured torque [N,6]")
    s = np.sqrt(np.mean(y*y, axis=0))
    ref = float(np.median(s))
    if ref <= 0:
        raise ValueError("All or most torque channels are zero; reference RMS vanishes")
    return 1.0/(s*s + ref*ref)


def solve_wls(Yb, tau, weights=None, rcond=1e-12):
    Yb, tau = np.asarray(Yb, float), np.asarray(tau, float)
    if Yb.ndim != 3 or Yb.shape[1] != 6 or tau.shape != Yb.shape[:2]:
        raise ValueError("Yb [N,6,p] and tau [N,6]")
    w = channel_weights(tau) if weights is None else np.asarray(weights, float)
    if w.shape != (6,) or np.any(w <= 0) or not np.isfinite(w).all():
        raise ValueError("Weights must be positive [6]")
    root = np.sqrt(w)[None, :, None]
    lhs = (Yb * root).reshape(-1, Yb.shape[-1])
    rhs = (tau * np.sqrt(w)[None, :]).ravel()
    coefficient, _, rank, _ = lstsq(lhs, rhs, cond=rcond, lapack_driver="gelsd")
    pred = np.einsum("njp,p->nj", Yb, coefficient)
    residual = pred - tau
    wrmse = float(np.sqrt(np.mean((residual* np.sqrt(w)[None, :])**2)))
    return Fit(coefficient, pred, residual, w, wrmse, int(rank))
