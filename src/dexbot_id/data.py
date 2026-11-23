"""Six-actuator trajectory and optional actuator-torque data interfaces."""
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import numpy as np

ACTUATORS = ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'rocker_left', 'rocker_right')
KINEMATIC_FIELDS = ['time'] + [f'{kind}{i}' for kind in ('q', 'v', 'vd') for i in range(6)]
TORQUE_FIELDS = [f'tau{i}' for i in range(6)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


@dataclass
class Samples:
    time: np.ndarray
    q: np.ndarray
    v: np.ndarray
    a: np.ndarray
    tau: np.ndarray = None

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        n = len(self.time)
        for key in ('q', 'v', 'a'):
            array = np.asarray(getattr(self, key), dtype=float)
            if array.shape != (n, 6) or not np.isfinite(array).all():
                raise ValueError(f'{key} must be finite [N,6]')
            setattr(self, key, array)
        if self.tau is not None:
            self.tau = np.asarray(self.tau, dtype=float)
            if self.tau.shape != (n, 6) or not np.isfinite(self.tau).all():
                raise ValueError('tau must be finite [N,6]')
        if n < 3 or not np.isfinite(self.time).all() or not np.all(np.diff(self.time) > 0):
            raise ValueError('Require N>=3 finite, strictly increasing timestamps')

    def take(self, indices):
        indices = np.asarray(indices, dtype=int)
        return Samples(self.time[indices], self.q[indices], self.v[indices],
                       self.a[indices], None if self.tau is None else self.tau[indices])

    def with_tau(self, tau):
        return Samples(self.time, self.q, self.v, self.a, tau)


def _read_rows(path, required):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in required):
            raise ValueError(f'CSV missing required fields: {required}')
        records = []
        for lineno, row in enumerate(reader, start=2):
            if not any((row.get(field) or '').strip() for field in required):
                continue
            try:
                records.append([float(row[field]) for field in required])
            except (TypeError, ValueError) as exc:
                raise ValueError(f'Invalid numeric CSV data at line {lineno}') from exc
    if not records:
        raise ValueError('CSV contains no records')
    output = np.asarray(records, dtype=float)
    if not np.isfinite(output).all():
        raise ValueError('Nonfinite numeric CSV data')
    return output


def load_id_csv(path):
    """Read 15-s or longer q/v/vd trajectory; torque columns are optional."""
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        fields = csv.DictReader(stream).fieldnames or []
    present = [key in fields for key in TORQUE_FIELDS]
    if any(present) and not all(present):
        raise ValueError('Provide all tau0..tau5 or omit the entire torque block')
    raw = _read_rows(path, KINEMATIC_FIELDS + (TORQUE_FIELDS if all(present) else []))
    return Samples(raw[:, 0], raw[:, 1:7], raw[:, 7:13], raw[:, 13:19],
                   raw[:, 19:25] if all(present) else None)


def load_torque_csv(path, expected_time=None):
    """Read calibrated drive torques in a separate CSV: time,tau0..tau5."""
    raw = _read_rows(path, ['time'] + TORQUE_FIELDS)
    time, tau = raw[:, 0], raw[:, 1:7]
    if not np.all(np.diff(time) > 0):
        raise ValueError('Torque timestamps must increase strictly')
    if expected_time is not None:
        reference = np.asarray(expected_time, dtype=float)
        if time.shape != reference.shape or not np.allclose(time, reference, rtol=0, atol=1e-7):
            raise ValueError('Torque data time grid does not match trajectory CSV')
    return time, tau


def paper_protocol_summary(data, dt=0.002, period=15.0):
    steps = np.diff(data.time)
    n = len(data.time)
    is_grid = (abs(data.time[0]) < 1e-8 and np.max(np.abs(steps - dt)) < 1e-7)
    return {
        'rows': n,
        'start_s': float(data.time[0]),
        'end_s': float(data.time[-1]),
        'median_dt_s': float(np.median(steps)),
        'fifteen_second_grid': bool(is_grid and n in (7500, 7501) and
                                    abs(data.time[-1] - (n - 1) * dt) < 1e-7),
        'four_period_grid': bool(is_grid and n == 30001 and
                                 abs(data.time[-1] - 4 * period) < 1e-7),
        'has_torque': data.tau is not None,
        'joint_order': list(ACTUATORS),
    }
