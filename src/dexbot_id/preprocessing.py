"""Paper §4.4: fourth-order velocity differentiation and zero-phase filtering."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


def fourth_order_acceleration(v, dt):
    v = np.asarray(v, float)
    if v.ndim != 2 or v.shape[0] < 5 or dt <= 0:
        raise ValueError("v must have at least five time rows and dt positive")
    a = np.full_like(v, np.nan)
    # Paper Eq. (29); first/last 2 samples are explicitly unavailable.
    a[2:-2] = (-v[4:] + 8*v[3:-1] - 8*v[1:-3] + v[:-4]) / (12*dt)
    return a


def zero_phase_filter(x, sample_hz, cutoff_hz, order=4, axis=0):
    if not (0 < cutoff_hz < sample_hz/2):
        raise ValueError("Cutoff must be between 0 and Nyquist frequency")
    sos = butter(order, cutoff_hz, fs=sample_hz, btype="low", output="sos")
    return sosfiltfilt(sos, np.asarray(x, float), axis=axis)
