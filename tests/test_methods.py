"""Equation-level tests on synthetic arrays; no claims about measured Dexbot robot."""
import numpy as np
import pytest
from dexbot_id.fourier import evaluate
from dexbot_id.projection import tangent_map, full_kinematics, project_full_regressor
from dexbot_id.preprocessing import fourth_order_acceleration, zero_phase_filter
from dexbot_id.regression import augmented_regressor, friction_block, stack
from dexbot_id.base import select_base_columns, base_to_full_prediction
from dexbot_id.wls import channel_weights, solve_wls
from dexbot_id.physical import pseudo_inertia_4x4, nominal_link_parameters
from dexbot_id.validation import rmse_by_actuator


def test_fourier_derivatives_and_periodicity():
    t=np.linspace(0,15,701)
    aa=np.ones((6,5))*.07;bb=np.ones((6,5))*.03
    q,v,a=evaluate(t,aa,bb,np.zeros(6))
    assert np.max(np.abs(q[0]-q[-1]))<1e-12
    assert np.max(np.abs(v[0]-v[-1]))<1e-12
    i=211; h=1e-6
    qp=evaluate(np.array([t[i]+h]),aa,bb,np.zeros(6))[0][0]
    qm=evaluate(np.array([t[i]-h]),aa,bb,np.zeros(6))[0][0]
    assert np.max(np.abs((qp-qm)/(2*h)-v[i]))<1e-8
    vp=evaluate(np.array([t[i]+h]),aa,bb,np.zeros(6))[1][0]
    vm=evaluate(np.array([t[i]-h]),aa,bb,np.zeros(6))[1][0]
    assert np.max(np.abs((vp-vm)/(2*h)-a[i]))<1e-8


def test_constraint_projection_and_acceleration():
    Ja=np.array([[1.,2.],[3.,4.]])
    Ju=np.array([[2.,0.],[0.,3.]])
    curvature=np.array([.5,-.3])
    G,bias=tangent_map(Ja,Ju,curvature)
    assert np.max(abs(np.c_[Ja,Ju]@G))<1e-14
    v,a=full_kinematics(G,bias,np.array([.2,.4]),np.array([.9,-.1]))
    assert np.max(abs(np.c_[Ja,Ju]@a+curvature))<1e-14
    Y=np.arange(20.).reshape(4,5)
    assert np.allclose(project_full_regressor(Y,G),G.T@Y)


def test_friction_blocks_exact():
    v=np.zeros((2,6));v[0,0]=.5;v[1,0]=-1.0
    a=np.ones((2,6))*3.
    phi=friction_block(v,a)
    assert phi.shape==(2,6,24)
    assert np.allclose(phi[0,0,:4],[1,.5,3,1])
    assert np.allclose(phi[1,0,:4],[-1,-1,3,1])
    assert phi[0,1,0]==0.
    assert np.allclose(phi[0,1,4:8],[0,0,3,1])
    Y=augmented_regressor(np.zeros((2,6,60)),v,a)
    assert Y.shape==(2,6,84)
    W,T=stack(Y,np.ones((2,6)));assert W.shape==(12,84) and T.shape==(12,)


def test_QR_base_mapping_exact_rank_deficiency():
    rng=np.random.default_rng(7)
    A=rng.normal(size=(60,3))
    W=np.c_[A,A[:,0]+2*A[:,2],-A[:,1],np.zeros(60)]
    info=select_base_columns(W)
    assert info.full_rank==3
    assert info.relative_reconstruction<1e-12
    theta=rng.normal(size=6)
    assert np.allclose(base_to_full_prediction(W,theta,info),W@theta,atol=1e-11)


def test_wls_weights_and_torque_fit():
    rng=np.random.default_rng(10)
    Y=rng.normal(size=(95,6,8));theta=np.arange(1,9)*.02
    t=np.einsum('njp,p->nj',Y,theta)
    fit=solve_wls(Y,t)
    assert np.max(np.abs(fit.coefficients-theta))<1e-11
    s=np.sqrt(np.mean(t*t,axis=0));ref=np.median(s)
    assert np.allclose(channel_weights(t),1/(s*s+ref*ref))
    assert np.max(rmse_by_actuator(fit.predictions,t))<1e-12


def test_fd4_on_quartic_velocity():
    t=np.arange(30)*.002
    v=np.stack([t**4+2*t**2 for _ in range(6)],axis=1)
    out=fourth_order_acceleration(v,.002)
    expected=4*t**3+4*t
    assert np.max(abs(out[2:-2,0]-expected[2:-2]))<1e-11
    assert np.isnan(out[:2]).all() and np.isnan(out[-2:]).all()


def test_filter_does_not_change_constant_signal():
    x=np.ones((300,6))*2
    assert np.allclose(zero_phase_filter(x,500.,25),x,atol=1e-10)


def test_pseudo_inertia_matrix_is_physical_for_simple_link():
    theta=nominal_link_parameters(m=2,com=(.01,-.02,.03),I_com=(.1,.13,.17))
    P=pseudo_inertia_4x4(theta)
    assert np.min(np.linalg.eigvalsh(P))>0
    assert np.isclose(P[-1,-1],2)


def test_validation_requires_same_dimensions():
    with pytest.raises(ValueError):rmse_by_actuator(np.ones((3,6)),np.ones((4,6)))


def test_singular_passive_jacobian_rejected():
    with pytest.raises(ValueError):tangent_map(np.ones((2,6)),np.zeros((2,2)))


def test_fourier_optimizer_rejects_missing_robot_joint_bounds():
    from dexbot_id.excitation_optimization import optimize_fourier
    with pytest.raises(ValueError):
        optimize_fourier(np.zeros((6,5)),np.zeros((6,5)),np.zeros(6),np.linspace(0,15,31),
                         lambda q,v,a: np.ones((30,2)),lower=np.array([-2.,-2.]),
                         upper=np.ones(6),velocity_bounds=np.ones(6),acceleration_bounds=np.ones(6))


def test_physical_LMI_does_not_silently_fallback():
    from dexbot_id.physical import solve_lmi_wls
    from importlib.util import find_spec
    if find_spec('cvxpy') is None:
        with pytest.raises(RuntimeError,match='requires'):
            solve_lmi_wls(np.zeros((3,6,34)),np.ones((3,6)),n_links=1)
