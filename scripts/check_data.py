#!/usr/bin/env python3
"""Validate the six-actuator identification CSV and optional matching inertia NPZ.

Does not modify source data or require Drake, robot hardware, or a model exporter.
"""
import argparse
import json
from pathlib import Path

from dexbot_id.data import ACTUATORS, load_id_csv, paper_protocol_summary, sha256


def validate(csv_path, inertia_npz=None):
    samples = load_id_csv(csv_path)
    result = {
        'project': 'Dexbot-DynID',
        'csv_path': str(Path(csv_path).resolve()),
        'csv_sha256': sha256(csv_path),
        'csv': paper_protocol_summary(samples),
        'actuator_order': list(ACTUATORS),
        'projected_inertial_regressor_provided': bool(inertia_npz),
        'drake_needed_to_read_inputs': False,
    }
    if inertia_npz is not None:
        # Reuse the exact input alignment contract checked by the fitting CLI.
        from run_identification import read_regressor
        design, selected, names, _ = read_regressor(inertia_npz, samples)
        result['inertial_regressor'] = {
            'path': str(Path(inertia_npz).resolve()),
            'sha256': sha256(inertia_npz),
            'shape': list(design.shape),
            'selected_rows': len(selected.time),
            'inertial_parameter_columns': len(names),
            'augmented_parameter_columns': len(names) + 24,
            'matches_manuscript_84_column_shape': len(names) == 60,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, help='Trajectory CSV with six-actuator kinematics')
    parser.add_argument('--inertia-npz', help='Optional precomputed matching six-actuator inertial regressor')
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.csv, args.inertia_npz), indent=2, ensure_ascii=False))
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, 'DATA VALIDATION ERROR: %s\n' % exc)


if __name__ == '__main__':
    main()
