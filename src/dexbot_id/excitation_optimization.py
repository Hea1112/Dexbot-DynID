"""Optional reference optimizer for manuscript Eq. (28), with user-supplied Y.

Reproducing the original optimized trajectory requires actual robot constraints
and a physically grounded regressor builder; no original coefficients or
workspace margins are invented here.
"""
import numpy as np
from scipy.optimize import minimize
from .fourier import evaluate,condition_number


def optimize_fourier(initial_a, initial_b, initial_offsets, times,
                     regressor_builder, lower, upper, velocity_bounds, acceleration_bounds,
                     workspace_inequality=None, maxiter=200):
    """SLSQP optimization of the Fourier coefficients with sampled constraints.

    `regressor_builder(q,v,a)` must return actual stacked Y_b at each iteration.
    User `workspace_inequality(q)` returns signed feasible margins >=0.
    As in the paper, boundary q(0)=v(0)=0. Periodic endpoints match by construction.
    Returns SciPy OptimizeResult and optimized (a,b,offsets). Not a guaranteed
    globally optimal excitation trajectory; workspace checks are sampled.
    """
    a=np.asarray(initial_a,float);b=np.asarray(initial_b,float)
    q0=np.asarray(initial_offsets,float);times=np.asarray(times,float)
    if a.shape!=b.shape or a.ndim!=2 or a.shape[0]!=6 or q0.shape!=(6,):
        raise ValueError('Expected initial a,b [6,K] and offsets [6]')
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    vmax=np.asarray(velocity_bounds,float);amax=np.asarray(acceleration_bounds,float)
    for x in (lower,upper,vmax,amax):
        if x.shape!=(6,):raise ValueError('Joint bounds must be [6]')
    if np.any(lower>=upper) or np.any(vmax<=0) or np.any(amax<=0):
        raise ValueError('Invalid constraints')
    n=a.size
    def unpack(x):return x[:n].reshape(a.shape),x[n:2*n].reshape(b.shape),x[2*n:2*n+6]
    def data(x):return evaluate(times,*unpack(x))
    def objective(x):
        q,v,acc=data(x)
        Y=np.asarray(regressor_builder(q,v,acc),float)
        if Y.ndim==3:Y=Y.reshape(-1,Y.shape[-1])
        c=condition_number(Y)
        return c if np.isfinite(c) else 1e15
    def margins(x):
        q,v,acc=data(x)
        parts=[(q-lower).ravel(),(upper-q).ravel(),(vmax-np.abs(v)).ravel(),(amax-np.abs(acc)).ravel()]
        if workspace_inequality is not None:
            parts.append(np.asarray(workspace_inequality(q),float).ravel())
        return np.concatenate(parts)
    def boundary(x):
        q,v,_=evaluate(np.array([0.]),*unpack(x))
        return np.r_[q.ravel(),v.ravel()]
    x0=np.r_[a.ravel(),b.ravel(),q0]
    result=minimize(objective,x0,method='SLSQP',constraints=[
        {'type':'ineq','fun':margins},{'type':'eq','fun':boundary}],
        options={'maxiter':int(maxiter),'ftol':1e-7})
    return result,unpack(result.x)
