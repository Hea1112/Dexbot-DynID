"""Production CLI contract tests; all arrays are generated in temporary test storage."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def create_inputs(folder, n=100, columns=60):
    t=np.arange(n)*0.002
    rng=np.random.default_rng(1337)
    q=rng.normal(scale=.05,size=(n,6))
    v=rng.normal(scale=.3,size=(n,6))
    a=rng.normal(scale=.4,size=(n,6))
    y=rng.normal(size=(n,6,columns))
    from dexbot_id.regression import augmented_regressor
    phi=augmented_regressor(y,v,a)
    theta=np.linspace(0.01,0.5,columns+24)
    tau=np.einsum('nij,j->ni',phi,theta)
    path=folder/'measurements.csv'
    header=['time']+[f'{k}{i}' for k in ('q','v','vd','tau') for i in range(6)]
    with path.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(header)
        for i in range(n):w.writerow([t[i],*q[i],*v[i],*a[i],*tau[i]])
    matrix=folder/'inertia.npz'
    np.savez(matrix,Y_in=y,time_s=t,paper_q6=q,paper_v6=v,paper_a6=a,
             parameter_names=np.array([f'param_{i}' for i in range(columns)]))
    return path,matrix


def invoke(root, csv_path, matrix_path, output):
    return subprocess.run([sys.executable,str(root/'scripts/run_identification.py'),
                           '--csv',str(csv_path),'--inertia-npz',str(matrix_path),
                           '--require-84','--data-kind','experiment-unverified',
                           '--output',str(output)],cwd=root,capture_output=True,text=True)


def test_production_cli_writes_method_results(tmp_path):
    root=Path(__file__).resolve().parents[1]
    csv_path,matrix_path=create_inputs(tmp_path)
    result=invoke(root,csv_path,matrix_path,tmp_path/'output')
    assert result.returncode==0,result.stderr
    rep=json.loads((tmp_path/'output/summary.json').read_text())
    assert rep['extended_regressor_shape']==[100,6,84]
    assert rep['extended_regressor_shape'][-1] == 84
    assert rep['computed_base_rank'] <= 84
    assert rep['relative_reconstruction_error'] < 1e-9
    assert rep['identification_status'] == 'WLS_completed'
    with np.load(tmp_path/'output/identification.npz',allow_pickle=False) as z:
        assert z['full_to_base_map'].shape[1]==84
        assert z['tau_observed'].shape==(100,6)


def test_rejects_non_manuscript_inertial_column_count(tmp_path):
    root=Path(__file__).resolve().parents[1]
    csv_path,matrix_path=create_inputs(tmp_path,columns=100)
    result=invoke(root,csv_path,matrix_path,tmp_path/'bad')
    assert result.returncode!=0
    assert 'requires 60 inertia columns' in result.stderr


def test_trajectory_without_torque_supports_basis_only(tmp_path):
    root=Path(__file__).resolve().parents[1]
    csv_path,matrix_path=create_inputs(tmp_path, n=100)
    no_tau=tmp_path/'trajectory.csv'
    with csv_path.open(newline='') as src, no_tau.open('w',newline='') as dst:
        rows=list(csv.reader(src))
        writer=csv.writer(dst)
        writer.writerows([line[:19] for line in rows])
    result=invoke(root,no_tau,matrix_path,tmp_path/'trajectory_out')
    assert result.returncode==0,result.stderr
    rep=json.loads((tmp_path/'trajectory_out/summary.json').read_text())
    assert rep['identification_status']=='basis_only'
    with np.load(tmp_path/'trajectory_out/identification.npz') as z:
        assert z['theta_base_WLS'].size == 0
        assert z['full_to_base_map'].shape[1] == 84


def test_separate_torque_csv_supports_wls(tmp_path):
    root=Path(__file__).resolve().parents[1]
    csv_path,matrix_path=create_inputs(tmp_path, n=100)
    motion=tmp_path/'trajectory.csv'
    torques=tmp_path/'torque.csv'
    with csv_path.open(newline='') as stream:
        lines=list(csv.reader(stream))
    with motion.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerows([line[:19] for line in lines])
    with torques.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time']+[f'tau{i}' for i in range(6)])
        writer.writerows([[line[0]]+line[19:25] for line in lines[1:]])
    result=subprocess.run([sys.executable,str(root/'scripts/run_identification.py'),
                           '--csv',str(motion),'--torque-csv',str(torques),
                           '--inertia-npz',str(matrix_path),'--require-84',
                           '--output',str(tmp_path/'separate_fit')],
                          cwd=root,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    rep=json.loads((tmp_path/'separate_fit/summary.json').read_text())
    assert rep['identification_status']=='WLS_completed'
    assert max(rep['WLS_training_rmse'].values()) < 1e-8


def test_separate_torque_grid_mismatch_rejected(tmp_path):
    root=Path(__file__).resolve().parents[1]
    csv_path,matrix_path=create_inputs(tmp_path, n=20)
    motion=tmp_path/'trajectory.csv'
    torques=tmp_path/'torque.csv'
    with csv_path.open(newline='') as stream: lines=list(csv.reader(stream))
    with motion.open('w',newline='') as stream: csv.writer(stream).writerows([line[:19] for line in lines])
    with torques.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['time']+[f'tau{i}' for i in range(6)])
        writer.writerows([[float(line[0])+0.01]+line[19:25] for line in lines[1:]])
    result=subprocess.run([sys.executable,str(root/'scripts/run_identification.py'),
                           '--csv',str(motion),'--torque-csv',str(torques),
                           '--inertia-npz',str(matrix_path),'--output',str(tmp_path/'bad')],
                          cwd=root,capture_output=True,text=True)
    assert result.returncode!=0 and 'time grid' in result.stderr
