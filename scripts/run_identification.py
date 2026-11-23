#!/usr/bin/env python3
"""CLI for manuscript-based QR, WLS, optional full-parameter LMI, and validation.

No input CSV, URDF or mount scenario is assumed to be public or physically
validated by this code. Output labels distinguish in-sample and independent.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dexbot_id.data import load_id_csv, load_torque_csv, paper_protocol_summary, sha256, ACTUATORS
from dexbot_id.regression import augmented_regressor, parameter_names
from dexbot_id.base import select_base_columns
from dexbot_id.wls import solve_wls
from dexbot_id.validation import rmse_by_actuator
from dexbot_id.physical import solve_lmi_wls


def read_regressor(path, samples):
    with np.load(path, allow_pickle=False) as z:
        if 'Y_in' in z:
            key = 'Y_in'
        elif 'Y' in z:
            key = 'Y'
        else:
            raise ValueError('NPZ requires Y_in or Y [N,6,p]')
        Y = np.asarray(z[key], float)
        if Y.ndim != 3 or Y.shape[1] != 6:
            raise ValueError('Regressor must be [N,6,p]')
        if 'source_indices' in z:
            ind = np.asarray(z['source_indices'], int)
            if len(np.unique(ind)) != len(ind) or len(ind) != len(Y):
                raise ValueError('Invalid source indices')
            if np.min(ind) < 0 or np.max(ind) >= len(samples.time):
                raise ValueError('source_indices out of range')
            selected = samples.take(ind)
        elif len(Y) == len(samples.time):
            selected = samples
        else:
            raise ValueError('Y rows differ from CSV; provide source_indices')
        tfield = 'time_s' if 'time_s' in z else ('time' if 'time' in z else None)
        if tfield is not None and np.max(np.abs(z[tfield] - selected.time)) > 1e-7:
            raise ValueError('Regressor time does not match CSV')
        for key, field in (('paper_q6','q'),('paper_v6','v'),('paper_a6','a')):
            if key in z and not np.allclose(z[key], getattr(selected,field),atol=1e-8,rtol=0):
                raise ValueError(f'Input {key} not aligned with CSV')
        names = [str(v) for v in z['parameter_names']] if 'parameter_names' in z else [f'inertia_{i}' for i in range(Y.shape[2])]
        nominal = np.asarray(z['theta_nominal'],float) if 'theta_nominal' in z else None
        if len(names) != Y.shape[2] or (nominal is not None and nominal.shape != (Y.shape[2],)):
            raise ValueError('Regressor metadata dimension mismatch')
    if not np.isfinite(Y).all():
        raise ValueError('Nonfinite regressor values')
    return Y, selected, names, nominal


def pipeline(args):
    raw = load_id_csv(args.csv)
    if args.torque_csv is not None:
        if raw.tau is not None:
            raise ValueError('Torque is already in --csv; do not also pass --torque-csv')
        _, torque = load_torque_csv(args.torque_csv, raw.time)
        raw = raw.with_tau(torque)
    Y_in, sampled, inertial_names, nominal = read_regressor(args.inertia_npz, raw)
    if args.require_84 and Y_in.shape[2] != 60:
        raise ValueError(f'84-column manuscript-shaped regressor requires 60 inertia columns; received {Y_in.shape[2]}')
    Y = augmented_regressor(Y_in, sampled.v, sampled.a, args.deadband)
    names = parameter_names(inertial_names)
    stacked = Y.reshape(-1, Y.shape[2])
    base = select_base_columns(stacked, rtol=args.rank_tolerance)
    if base.relative_reconstruction > args.reconstruction_tolerance:
        raise RuntimeError('QR basis reconstruction tolerance exceeded')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    fit = None
    full_lmi = None
    report = {
        'project': 'Dexbot-DynID',
        'dataset': paper_protocol_summary(raw),
        'samples': len(sampled.time),
        'inertial_regressor_shape': list(Y_in.shape),
        'extended_regressor_shape': list(Y.shape),
        'computed_base_rank': base.full_rank,
        'rank_tolerance': args.rank_tolerance,
        'relative_reconstruction_error': base.relative_reconstruction,
        'identification_status': 'basis_only' if sampled.tau is None else 'WLS_completed',
        'input_sha256': {'csv': sha256(args.csv), 'inertia_npz': sha256(args.inertia_npz)},
    }
    if args.torque_csv is not None:
        report['input_sha256']['torque_csv'] = sha256(args.torque_csv)
    if args.lmi and sampled.tau is None:
        raise ValueError('LMI fitting requires tau0..tau5 in --csv or --torque-csv')
    if sampled.tau is not None:
        fit = solve_wls(Y[:, :, base.selected], sampled.tau)
        report['WLS_rank'] = fit.rank
        report['WLS_weights'] = fit.channel_weights.tolist()
        report['WLS_weighted_rmse'] = fit.weighted_rmse
        report['WLS_training_rmse'] = dict(zip(ACTUATORS, rmse_by_actuator(fit.predictions, sampled.tau).tolist()))
        report['torque_unit'] = args.torque_unit
        if args.lmi:
            n_links = args.n_links
            if Y_in.shape[2] != 10 * n_links:
                raise ValueError('LMI requires 10 inertial columns per link')
            full_lmi = solve_lmi_wls(Y, sampled.tau, n_links, mapping=base,
                                     weights=fit.channel_weights, solver=args.sdp_solver,
                                     prior_strength=args.prior_strength)
            report['LMI_status'] = full_lmi['status']
            report['LMI_solver'] = full_lmi['solver']
            report['LMI_training_rmse'] = dict(zip(ACTUATORS, rmse_by_actuator(full_lmi['predictions'], sampled.tau).tolist()))
        if nominal is not None:
            model_prediction = np.einsum('njk,k->nj', Y, np.r_[nominal, np.zeros(24)])
            report['nominal_training_rmse'] = dict(zip(ACTUATORS, rmse_by_actuator(model_prediction, sampled.tau).tolist()))
    if (args.validation_csv is None) != (args.validation_inertia_npz is None):
        raise ValueError('Validation requires both --validation-csv and --validation-inertia-npz')
    if args.validation_torque_csv is not None and args.validation_csv is None:
        raise ValueError('--validation-torque-csv requires validation CSV and regressor')
    if args.validation_csv is not None:
        if fit is None:
            raise ValueError('Validation requires torque observations for identification')
        valraw = load_id_csv(args.validation_csv)
        if args.validation_torque_csv is not None:
            if valraw.tau is not None:
                raise ValueError('Validation torque appears in both CSV files')
            _, tau_val = load_torque_csv(args.validation_torque_csv, valraw.time)
            valraw = valraw.with_tau(tau_val)
        if valraw.tau is None:
            raise ValueError('Validation torque columns are missing')
        Yv_in, val, val_names, _ = read_regressor(args.validation_inertia_npz, valraw)
        if val_names != inertial_names:
            raise ValueError('Validation inertial parameter names/order do not match')
        Yv = augmented_regressor(Yv_in, val.v, val.a, args.deadband)
        predictions = np.einsum('njk,k->nj', Yv[:, :, base.selected], fit.coefficients)
        report['independent_validation_rmse'] = dict(zip(ACTUATORS, rmse_by_actuator(predictions, val.tau).tolist()))
        if full_lmi is not None:
            predictions_lmi = np.einsum('njk,k->nj', Yv, full_lmi['theta_full'])
            report['independent_validation_LMI_rmse'] = dict(zip(ACTUATORS, rmse_by_actuator(predictions_lmi, val.tau).tolist()))
        report['input_sha256']['validation_csv'] = sha256(args.validation_csv)
        report['input_sha256']['validation_inertia_npz'] = sha256(args.validation_inertia_npz)
        if args.validation_torque_csv is not None:
            report['input_sha256']['validation_torque_csv'] = sha256(args.validation_torque_csv)
    elif args.validation_torque_csv is not None:
        raise ValueError('--validation-torque-csv requires a validation trajectory')
    np.savez_compressed(output / 'identification.npz',
                        selected_columns=base.selected,
                        full_to_base_map=base.F,
                        theta_base_WLS=fit.coefficients if fit is not None else np.array([]),
                        theta_full_LMI=full_lmi['theta_full'] if full_lmi is not None else np.array([]),
                        tau_observed=sampled.tau if sampled.tau is not None else np.empty((0, 6)),
                        tau_predicted=fit.predictions if fit is not None else np.empty((0, 6)),
                        time_s=sampled.time,
                        parameter_names=np.asarray(names, dtype='U128'))
    (output / 'summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'selected_columns.csv').write_text('original_col_0based,name\n' + ''.join(f'{i},{names[i]}\n' for i in base.selected), encoding='utf-8')
    print('Dexbot-DynID:', report['identification_status'])
    print('Samples:', len(sampled.time), 'inertia columns:', Y_in.shape[2], 'augmented columns:', Y.shape[2], 'base rank:', base.full_rank)
    print('Output:', output)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True,help='Trajectory CSV: time,q0..q5,v0..v5,vd0..vd5; tau0..tau5 optional')
    p.add_argument('--torque-csv',help='Optional separate CSV: time,tau0..tau5')
    p.add_argument('--inertia-npz',required=True,help='NPZ with actuator-projected inertia regressor Y_in or Y and optional source_indices')
    p.add_argument('--output',default='runs/identification')
    p.add_argument('--torque-unit',choices=['Nm','A','unknown'],default='unknown',help='User-declared units; NO calibration performed')
    p.add_argument('--data-kind',choices=['experiment-unverified','experiment-user-declared'],default='experiment-unverified')
    p.add_argument('--require-84',action='store_true',help='Reject models not providing 60 inertial +24 friction columns')
    p.add_argument('--rank-tolerance',type=float,default=1e-10)
    p.add_argument('--reconstruction-tolerance',type=float,default=1e-6)
    p.add_argument('--deadband',type=float,default=0.)
    p.add_argument('--validation-csv')
    p.add_argument('--validation-inertia-npz')
    p.add_argument('--validation-torque-csv')
    p.add_argument('--lmi',action='store_true',help='Optional physical-consistency SDP; pip install -e .[sdp]')
    p.add_argument('--n-links',type=int,default=6,help='Number of link 10-parameter inertia blocks for LMI')
    p.add_argument('--sdp-solver',default=None)
    p.add_argument('--prior-strength',type=float,default=0.)
    args=p.parse_args()
    try:
        pipeline(args)
    except (ValueError,RuntimeError,KeyError,OSError) as exc:
        p.exit(2,f'INPUT OR IDENTIFICATION ERROR: {exc}\n')

if __name__=='__main__':main()
