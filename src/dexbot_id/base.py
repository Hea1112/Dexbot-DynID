"""Paper Eq. (26): pivoted QR and full-to-base linear dependency mapping."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import qr


@dataclass
class BaseMap:
    selected: np.ndarray
    F: np.ndarray
    singular_values_normalized: np.ndarray
    relative_reconstruction: float
    full_rank: int
    raw_column_norms: np.ndarray


def select_base_columns(A, rtol=1e-10):
    """Pivoted QR on column-normalized design, using an SVD rank decision.

    Returns F such that A ~= A[:,selected] @ F, and beta=F@theta.
    Rank is trajectory/threshold dependent; never forced to reported 59.
    """
    A = np.asarray(A, float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all() or not 0 < rtol < 1:
        raise ValueError("Expected finite nonempty matrix and 0 < rtol < 1")
    norms = np.linalg.norm(A, axis=0)
    positive = norms > (np.max(norms) * 1e-13)
    if not positive.any():
        raise ValueError("Design has no active columns")
    active = np.flatnonzero(positive)
    normA = A[:, active] / norms[active][None, :]
    singular = np.linalg.svd(normA, compute_uv=False)
    rank = int(np.sum(singular > singular[0] * rtol))
    _, _, pivot = qr(normA, pivoting=True, mode="economic")
    sel = active[pivot[:rank]]
    Ab = A[:, sel]
    # One stable least-squares projection for all original columns.
    F, _, _, _ = np.linalg.lstsq(Ab, A, rcond=rtol)
    rel = float(np.linalg.norm(Ab @ F - A, 'fro') / np.linalg.norm(A, 'fro'))
    return BaseMap(sel, F, singular, rel, rank, norms)


def base_to_full_prediction(A, theta, mapping):
    A, theta = np.asarray(A, float), np.asarray(theta, float)
    return A[:, mapping.selected] @ (mapping.F @ theta)
