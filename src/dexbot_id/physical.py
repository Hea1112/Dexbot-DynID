"""Paper Eqs. (32)-(34): physically feasible pseudo-inertia LMI.

Ten inertial parameters per moving link: [m,hx,hy,hz,
Ixx_origin,Iyy_origin,Izz_origin,Ixy_origin,Ixz_origin,Iyz_origin].
The inertia tensor MUST be about the link origin, not the COM. For origin
inertia L, P=[[tr(L)/2 I-L,h],[h^T,m]] is affine in physical parameters.
Constraints are applied in the ORIGINAL full-inertia parameter space,
not directly to a reduced base-parameter vector.
"""
import numpy as np


def pseudo_inertia_4x4(theta10):
    x = np.asarray(theta10, float)
    if x.shape != (10,):
        raise ValueError("Expected 10 link inertial parameters")
    m = x[0]; h = x[1:4]
    xx,yy,zz,xy,xz,yz = x[4:10]
    L = np.array([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    P = np.zeros((4,4))
    P[:3,:3] = np.trace(L)/2 * np.eye(3) - L
    P[:3,3] = h; P[3,:3] = h; P[3,3] = m
    return P


def nominal_link_parameters(m=1.0, com=(0.,0.,0.), I_com=(0.05,0.06,0.07)):
    com = np.asarray(com,float); Ic=np.diag(np.asarray(I_com,float))
    I0 = Ic + m*(com@com*np.eye(3)-np.outer(com,com))
    return np.array([m,*(m*com),I0[0,0],I0[1,1],I0[2,2],I0[0,1],I0[0,2],I0[1,2]])


def _cvx_link_pseudo_inertia(x, cp):
    xx,yy,zz,xy,xz,yz = (x[j] for j in range(4,10))
    L=cp.bmat([[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]])
    first=(xx+yy+zz)/2*cp.Constant(np.eye(3))-L
    h=cp.reshape(x[1:4],(3,1),order='F')
    return cp.bmat([[first,h],[h.T,cp.reshape(x[0],(1,1),order='F')]])


def solve_lmi_wls(Y_full, tau, n_links, mapping=None, weights=None,
                  mass_epsilon=1e-6, inertia_epsilon=1e-9,
                  solver=None, max_iters=None, prior=None, prior_strength=0.):
    """Solve full-parameter constrained Eq. (34) with optional cvxpy.

    Y_full: [N,6, 10*n_links+24]. If `mapping` is BaseMap, minimize on
    Yb @ (F @ full_theta), enforcing LMIs on full_theta. This accounts for
    the non-unique original inertial vector underlying base parameters.
    SDP solve availability/status is explicit; no fallback fake LMI.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError('LMI option requires: pip install -e ".[sdp]"') from exc
    from .wls import channel_weights
    Y_full=np.asarray(Y_full,float); tau=np.asarray(tau,float)
    p=10*int(n_links)+24
    if n_links < 1 or Y_full.ndim!=3 or Y_full.shape[1:]!=(6,p) or tau.shape != Y_full.shape[:2]:
        raise ValueError("Y_full must be [N,6,10*n_links+24]; tau [N,6]")
    if not np.isfinite(Y_full).all() or not np.isfinite(tau).all():
        raise ValueError("Nonfinite LMI data")
    if mass_epsilon <= 0 or inertia_epsilon <= 0:
        raise ValueError("Strict physical inequality must be relaxed with positive epsilon")
    w=channel_weights(tau) if weights is None else np.asarray(weights,float)
    if w.shape!=(6,) or np.any(w<=0):raise ValueError("Weights must be positive [6]")
    x=cp.Variable(p)
    cons=[]
    for i in range(n_links):
        t=x[10*i:10*i+10]
        cons += [t[0] >= mass_epsilon,
                 _cvx_link_pseudo_inertia(t,cp) >> inertia_epsilon*np.eye(4)]
    for j in range(6):
        o=10*n_links+4*j
        cons += [x[o]>=0, x[o+1]>=0, x[o+2]>=0]
    W=Y_full.reshape(-1,p)
    if mapping is not None:
        if mapping.F.shape[1]!=p:
            raise ValueError("Mapping width must equal full parameter dimension")
        W=W[:,mapping.selected]@mapping.F
    A=W*np.tile(np.sqrt(w), len(tau))[:,None]
    b=tau.reshape(-1)*np.tile(np.sqrt(w),len(tau))
    obj=cp.sum_squares(A@x-b)
    if prior_strength:
        if prior is None or len(prior)!=p:
            raise ValueError("Prior vector required for positive prior_strength")
        obj += float(prior_strength)*cp.sum_squares(x-np.asarray(prior,float))
    problem=cp.Problem(cp.Minimize(obj),cons)
    settings={}
    if max_iters is not None and solver=='SCS':settings['max_iters']=int(max_iters)
    problem.solve(solver=solver,**settings)
    if x.value is None or problem.status not in ('optimal','optimal_inaccurate'):
        raise RuntimeError('SDP optimization failed; status='+str(problem.status))
    pred=(W@x.value).reshape(-1,6)
    return dict(theta_full=np.asarray(x.value).ravel(),
                theta_base=mapping.F@x.value if mapping is not None else None,
                predictions=pred,status=str(problem.status),solver=str(problem.solver_stats.solver_name),
                objective=float(problem.value),weights=w)
