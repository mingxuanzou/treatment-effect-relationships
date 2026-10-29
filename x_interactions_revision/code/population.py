"""Exact population and orthogonalize function source from finite_v2.py."""
import numpy as np
from finite import finite_truth

def orthogonalize(A,X):
    return A-X@np.linalg.solve(X.T@X,X.T@A)

def population(s):
    n=s['n']; rng=np.random.default_rng(s['population_seed'])
    raw=rng.uniform(-np.sqrt(3),np.sqrt(3),(n,8))
    cols=[]
    for j in range(3):
        v=raw[:,j]-raw[:,j].mean()
        for old in cols: v-=old*(old@v)/(old@old)
        cols.append(v/v.std(ddof=1))
    x=np.array(cols).T; X=np.c_[np.ones(n),x]; U=raw[:,3:5]; irr=raw[:,5:]
    a=np.array([np.sqrt(.4),0.,np.sqrt(.6)])
    bperp=.8*np.array([-np.sqrt(.6),0.,np.sqrt(.4)])+.6*np.array([0.,1.,0.])
    b=s['rho']*a+np.sqrt(1-s['rho']**2)*bperp
    sd1,sd2=.8,1.1
    if s['weak']=='both': sd1,sd2=s['strength'],1.375*s['strength']
    elif s['weak']=='second': sd2=s['strength']
    eps=rng.normal(size=(n,4))
    nonlinear=np.c_[x[:,0]**2-1,x[:,1]**2-1]
    nonlinear/=nonlinear.std(axis=0,ddof=1)
    if s['schedule']=='shared':
        shared=np.c_[U[:,0],U[:,0]]
        noise=np.c_[eps[:,0],.4*eps[:,0]+np.sqrt(.84)*eps[:,1]]
        base=.8*shared+.3*nonlinear+.5*noise
    else:
        base=.8*U+.3*nonlinear+.5*eps[:,:2]
    # Realized residual SD exactly one before choosing regular-point noise scale.
    base=orthogonalize(base,X); base/=base.std(axis=0,ddof=1)
    noise_scale=.45 if s['schedule']=='low_noise' and s['weak']=='none' else 1.
    Y0=np.c_[.25*x[:,0],-.25*x[:,1]]+noise_scale*base
    h=orthogonalize(eps[:,2:],X); h/=h.std(axis=0,ddof=1)
    tau=np.c_[1+sd1*(x@a),-.5+sd2*(x@b)]+s['kappa']*h
    Y1=Y0+tau
    # Expected quadratic inflation in projected moment estimators from the
    # first-order coefficient covariance; explicitly records the mechanism.
    tt=finite_truth(X,Y0,Y1,s['p'])
    K=X.shape[1]; S=np.cov(X,rowvar=False,ddof=1); C=tt['ols']['beta_sigma']/n
    diagnostics=dict(projected_noise_sd=noise_scale,
        baseline_residual_correlation=float(np.corrcoef(base.T)[0,1]),
        quadratic_noise_V11=float(np.trace(S@C[:K,:K])),
        quadratic_noise_V22=float(np.trace(S@C[K:,K:])),
        quadratic_noise_V12=float(np.trace(S@C[:K,K:])))
    return X,U,irr,Y0,Y1,diagnostics
