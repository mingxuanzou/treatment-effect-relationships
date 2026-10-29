"""Finite-population estimators in Zou draft, equations hat-beta-OLS/finite-adjusted-beta.
All matrices beta_cov are on ordinary variance scale; sigma2/gap in finite_truth
are on sqrt(n) scale. Outcomes are stacked outcome-major in score matrices.
"""
import numpy as np
from scipy.stats import norm

NAMES=['ATE1','ATE2','V11','V22','V12','rho','theta','alpha']
CRITICAL=float(norm.ppf(.975))

def covariance(A):
    A=np.asarray(A,float); C=A-A.mean(axis=0)
    return C.T@C/(len(A)-1)

def moments(X,beta):
    X=np.asarray(X,float); beta=np.asarray(beta,float)
    K=X.shape[1]; M=X.mean(axis=0); S=covariance(X); Sb=S@beta
    V=beta.T@Sb; a=M@beta
    v11,v22,v12=V[0,0],V[1,1],V[0,1]
    rho=v12/np.sqrt(v11*v22) if v11>0 and v22>0 else np.nan
    theta=v12/v11 if v11>0 else np.nan
    alpha=a[1]-theta*a[0]
    est=np.array([*a,v11,v22,v12,rho,theta,alpha])
    G=np.zeros((8,2*K)); G[0,:K]=M; G[1,K:]=M
    G[2,:K]=2*Sb[:,0]; G[3,K:]=2*Sb[:,1]
    G[4,:K]=Sb[:,1]; G[4,K:]=Sb[:,0]
    if v11>0 and v22>0:
        G[5]=G[4]/np.sqrt(v11*v22)-rho*G[2]/(2*v11)-rho*G[3]/(2*v22)
    else: G[5]=np.nan
    if v11>0:
        G[6]=(G[4]-theta*G[2])/v11
        G[7]=G[1]-theta*G[0]-a[0]*G[6]
    else: G[6:]=np.nan
    return est,G

def _result(X,beta,C):
    estimate,G=moments(X,beta)
    var=np.einsum('ij,jk,ik->i',G,C,G)
    # A negative computed quadratic form is a failure, never silently clipped.
    se=np.sqrt(np.where(var>=0,var,np.nan))
    return dict(names=NAMES.copy(),estimate=estimate,se=se,
                lower=estimate-CRITICAL*se,upper=estimate+CRITICAL*se,
                beta=beta,beta_cov=C,gradient=G,
                estimate_success=np.isfinite(estimate),
                interval_success=np.isfinite(estimate)&np.isfinite(se)&(se>0))

def fit_finite(X,Y,Z,W=None):
    """Fit both observed outcomes jointly, retaining cross-outcome score covariance."""
    X=np.asarray(X,float); Y=np.asarray(Y,float); Z=np.asarray(Z,int)
    n,K=X.shape
    if Y.shape!=(n,2) or Z.shape!=(n,): raise ValueError('invalid shapes')
    if W is not None: W=np.asarray(W,float); wbar=W.mean(axis=0)
    gam=[]; C=np.zeros((2*K,2*K)); Cm=C.copy(); corrections=[]
    for z in (0,1):
        keep=Z==z; x=X[keep]; y=Y[keep]; nz=len(x)
        if nz<=K or np.linalg.matrix_rank(x)<K: raise np.linalg.LinAlgError('singular X arm')
        S_inv=np.linalg.inv(x.T@x/nz)
        g=S_inv@(x.T@y/nz); gam.append(g)
        resid=y-x@g
        R=np.concatenate([x*resid[:,j,None] for j in range(2)],axis=1)
        A=np.kron(np.eye(2),S_inv)
        C+=A@(R.T@R/(nz-1))@A.T/nz
        if W is not None:
            w=W[keep]
            if nz<=w.shape[1] or np.linalg.matrix_rank(w)<w.shape[1]: raise np.linalg.LinAlgError('singular W arm')
            D=np.linalg.solve(w.T@w,w.T@R)
            E=R-w@D
            Cm+=A@(E.T@E/(nz-1))@A.T/nz
            corrections.append((A@(D.T@(w.mean(axis=0)-wbar))).reshape(2,K).T)
    beta=gam[1]-gam[0]
    out={'ols':_result(X,beta,C)}
    if W is not None:
        out['model_assisted']=_result(X,beta-corrections[1]+corrections[0],Cm)
    return out

def finite_truth(X,Y0,Y1,p=.5,W=None):
    """Exact finite-n analogues of asymptotic variance and conservative gap."""
    X=np.asarray(X,float); n,K=X.shape
    Sinv=np.linalg.inv(X.T@X/n); A=np.kron(np.eye(2),Sinv)
    R=[]; gam=[]
    for Y in (Y0,Y1):
        g=Sinv@(X.T@Y/n); gam.append(g)
        e=Y-X@g
        R.append(np.concatenate([X*e[:,j,None] for j in range(2)],axis=1))
    beta=gam[1]-gam[0]; est,G=moments(X,beta)
    sets={'ols':R}
    if W is not None:
        W=np.asarray(W,float)
        sets['model_assisted']=[r-W@np.linalg.solve(W.T@W,W.T@r) for r in R]
    out={}
    for key,(r0,r1) in sets.items():
        gap=A@covariance(r1-r0)@A.T
        con=A@(covariance(r1)/p+covariance(r0)/(1-p))@A.T
        C=con-gap
        out[key]=dict(names=NAMES.copy(),estimate=est,beta=beta,gradient=G,
                      sigma2=np.einsum('ij,jk,ik->i',G,C,G),
                      gap=np.einsum('ij,jk,ik->i',G,gap,G),
                      beta_sigma=C,beta_gap=gap)
    return out

def unbiased_coefficients(X,Y,Z):
    n=len(X); S=X.T@X/n
    return np.linalg.solve(S,X[Z==1].T@Y[Z==1]/sum(Z==1)-X[Z==0].T@Y[Z==0]/sum(Z==0))
