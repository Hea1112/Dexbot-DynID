"""Paper Eq. (27): K-harmonic finite Fourier excitation reference."""
import numpy as np


def evaluate(t, a, b, q0, period=15.):
    """a,b have shape (6,K); q0 shape (6,); outputs q,qd,qdd [N,6]."""
    t = np.atleast_1d(np.asarray(t, float))
    a, b, q0 = np.asarray(a, float), np.asarray(b, float), np.asarray(q0, float)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] != 6 or q0.shape != (6,) or period <= 0:
        raise ValueError("Expected coefficients [6,K], offsets [6], and positive period")
    k = np.arange(1, a.shape[1] + 1, dtype=float)
    wk = k * (2 * np.pi / period)
    angle = t[:, None] * wk[None, :]
    sn, cs = np.sin(angle), np.cos(angle)
    q = q0[None, :] + (sn @ (a / wk).T) - (cs @ (b / wk).T)
    v = cs @ a.T + sn @ b.T
    acc = -sn @ (a * wk).T + cs @ (b * wk).T
    return q, v, acc


def condition_number(Y):
    """Eq. (28) objective; rejects rank-deficient observation matrices."""
    s = np.linalg.svd(np.asarray(Y, float), compute_uv=False)
    return float(s[0] / s[-1]) if s.size and s[-1] > 0 else float('inf')
