"""Checks for actual-input inspection CLI. No distributed test datasets."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/check_data.py'


def fixture(tmp_path):
    f = tmp_path / 'trajectory.csv'
    headers = ['time'] + [f'{name}{j}' for name in ('q', 'v', 'vd', 'tau') for j in range(6)]
    with f.open('w', newline='') as output:
        w = csv.writer(output)
        w.writerow(headers)
        for i in range(5):
            w.writerow([i * .002] + [0.01 * (i + 1)] * 24)
    return f


def invoke(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], cwd=ROOT,
                          text=True, capture_output=True)


def test_csv_only_does_not_claim_complete_fit(tmp_path):
    f = fixture(tmp_path)
    p = invoke('--csv', f)
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data['csv']['rows'] == 5
    assert data['projected_inertial_regressor_provided'] is False
    assert data['drake_needed_to_read_inputs'] is False


def test_regressor_metadata_and_shape(tmp_path):
    f = fixture(tmp_path)
    pth = tmp_path / 'Y60.npz'
    np.savez_compressed(pth, Y_in=np.zeros((5, 6, 60)), time_s=np.arange(5)*.002)
    p = invoke('--csv', f, '--inertia-npz', pth)
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    assert data['inertial_regressor']['shape'] == [5, 6, 60]
    assert data['inertial_regressor']['matches_manuscript_84_column_shape'] is True


def test_mismatched_regressor_rejected(tmp_path):
    f = fixture(tmp_path)
    pth = tmp_path / 'bad.npz'
    np.savez_compressed(pth, Y_in=np.zeros((4, 6, 60)))
    p = invoke('--csv', f, '--inertia-npz', pth)
    assert p.returncode != 0
    assert 'source_indices' in p.stderr


def test_csv_can_be_only_one_15_second_cycle_without_torque(tmp_path):
    filename=tmp_path/'id_15s.csv'
    keys=['time'] + [f'{name}{i}' for name in ('q','v','vd') for i in range(6)]
    with filename.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(keys)
        for idx in range(7501):
            writer.writerow([idx * 0.002] + [0.0] * 18)
    proc=invoke('--csv',filename)
    assert proc.returncode==0,proc.stderr
    result=json.loads(proc.stdout)
    assert result['csv']['fifteen_second_grid'] is True
    assert result['csv']['has_torque'] is False
    assert result['csv']['rows']==7501
