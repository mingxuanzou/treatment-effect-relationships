"""Cross-fitted reduced-CATE relationship estimation from the supplied draft.
Componentwise least-squares gradient boosting of binary regression stumps.
The Gram-matrix implementation is algebraically the same as refitting stumps
at each boosting round; no outcome-dependent tuning, selection, or clipping of CIs.
"""
import numpy as np
from scipy.stats import norm
NAMES=['ATE1','ATE2','V11','V22','V12','rho','theta','alpha']
ZCRIT=norm.ppf(.975)

class StumpBoost:
    def __init__(self,n_estimators=500,learning_rate=.1,max_bins=None):
        self.n_estimators=n_estimators;self.learning_rate=learning_rate;self.max_bins=max_bins
    def _design(self,X):
        return np.column_stack([(X[:,j]>t).astype(float) for j,t in self.cuts]) if self.cuts else np.zeros((len(X),0))
    def fit(self,X,Y):
        X=np.asarray(X,float);Y=np.asarray(Y,float)
        self.scalar=Y.ndim==1
        if self.scalar:Y=Y[:,None]
        bins=self.max_bins or min(24,max(4,int(len(X)**.32)))
        self.cuts=[]
        for j in range(X.shape[1]):
            u=np.unique(X[:,j])
            ts=(u[:-1]+u[1:])/2 if len(u)<=bins else np.unique(np.quantile(X[:,j],np.arange(1,bins)/bins))
            self.cuts.extend((j,float(t)) for t in ts)
        H=self._design(X);self.hmean=H.mean(0);H-=self.hmean
        self.intercept=Y.mean(0);self.coef=np.zeros((H.shape[1],Y.shape[1]))
        if H.shape[1]:
            G=H.T@H;cc=H.T@(Y-self.intercept);d=np.diag(G);d=np.maximum(d,1e-12)
            for _ in range(self.n_estimators):
                idx=np.argmax(cc*cc/d[:,None],axis=0)
                step=self.learning_rate*cc[idx,np.arange(Y.shape[1])]/d[idx]
                self.coef[idx,np.arange(Y.shape[1])]+=step
                cc-=G[:,idx]*step
        return self
    def predict(self,X):
        y=self.intercept+(self._design(np.asarray(X))-self.hmean)@self.coef
        return y[:,0] if self.scalar else y

def _moments(P,T,center=None):
    n=len(P);a=P.mean(0) if center is None else np.asarray(center)
    tc=T-a;r=P-T
    components=np.column_stack((tc[:,0]**2+2*tc[:,0]*r[:,0],tc[:,1]**2+2*tc[:,1]*r[:,1],tc[:,0]*tc[:,1]+tc[:,1]*r[:,0]+tc[:,0]*r[:,1]))
    v=components.mean(0)
    return a,v,components-v

def _result(a,v,ifa,ifv,naive=False):
    n=len(ifa);v1,v2,v12=v
    est=np.r_[a,v,[np.nan]*3];eif=np.column_stack((ifa,ifv,np.full((n,3),np.nan)))
    if v1>0 and v2>0:
        rho=v12/np.sqrt(v1*v2);theta=v12/v1;alpha=a[1]-theta*a[0]
        est[5:]=[rho,theta,alpha]
        eif[:,5]=ifv[:,2]/np.sqrt(v1*v2)-rho*ifv[:,0]/(2*v1)-rho*ifv[:,1]/(2*v2)
        eif[:,6]=(ifv[:,2]-theta*ifv[:,0])/v1
        eif[:,7]=ifa[:,1]-theta*ifa[:,0]-a[0]*eif[:,6]
    se=np.sqrt(np.mean(eif*eif,axis=0)/n)
    if naive and np.isfinite(est[6]):
        # Usual homoskedastic OLS SEs for fitted-CATE regression, not EIF SEs.
        u=ifa[:,1]-est[6]*ifa[:,0]
        rv=np.dot(u,u)/max(n-2,1)
        se[6]=np.sqrt(rv/(n*v1));se[7]=np.sqrt(rv*(1+a[0]**2/v1)/n)
    return dict(names=NAMES,estimate=est,se=se,lower=est-ZCRIT*se,upper=est+ZCRIT*se,eif=eif)

def from_predictions(P,T,h=None):
    P=np.asarray(P);T=np.asarray(T);a,v,ifv=_moments(P,T)
    direct=_result(a,v,P-a,ifv)
    H=np.column_stack((np.ones(len(P)),T));coef=np.linalg.lstsq(H,P,rcond=None)[0];C=H@coef
    ac,vc,ifvc=_moments(P,C);cal=_result(ac,vc,P-ac,ifvc)
    at=T.mean(0);vn=np.array([np.var(T[:,0]),np.var(T[:,1]),np.mean((T[:,0]-at[0])*(T[:,1]-at[1]))])
    tf=T-at;nf=np.column_stack((tf[:,0]**2-vn[0],tf[:,1]**2-vn[1],tf[:,0]*tf[:,1]-vn[2]))
    naive=_result(at,vn,tf,nf,naive=True)
    _,vs,_=_moments(P,T,center=at)
    mat=lambda q:np.array([[q[0],q[2]],[q[2],q[1]]])
    diag=dict(condition_number=float(np.linalg.cond(H.T@H/len(P))),direct_min_eig=float(np.linalg.eigvalsh(mat(v)).min()),min_eig_sample_minus_direct=float(np.linalg.eigvalsh(mat(vs-v)).min()),min_eig_cal_minus_sample=float(np.linalg.eigvalsh(mat(vc-vs)).min()),direct_negative_v1=bool(v[0]<0),direct_negative_v2=bool(v[1]<0),direct_rho_outside=bool(np.isfinite(direct['estimate'][5]) and abs(direct['estimate'][5])>1),cal_mean_error=float(np.max(abs(C.mean(0)-a))),sample_gap_v1=vs[0]-v[0],sample_gap_v2=vs[1]-v[1],cal_gap_v1=vc[0]-v[0],cal_gap_v2=vc[1]-v[1])
    results={'direct':direct,'calibrated':cal,'naive':naive,'diagnostics':diag,'pseudo':P,'tau':T,'calibrated_tau':C,'sample_centered_cov':vs}
    if h is not None:
        h=np.asarray(h);h=h[:,None] if h.ndim==1 else h
        HH=np.column_stack((H,h));CH=HH@np.linalg.lstsq(HH,P,rcond=None)[0]
        diag['h_condition_number']=float(np.linalg.cond(HH.T@HH/len(P)));diag['h_rank']=int(np.linalg.matrix_rank(HH))
        ah,vh,ifvh=_moments(P,CH);results['calibrated_h']=_result(ah,vh,P-ah,ifvh)
        diag['h_normal_equation_error']=float(np.max(abs(HH.T@(P-CH))/len(P)))
        if h.shape[1]==1 and set(np.unique(h)).issubset({0,1}):
            mask=h[:,0].astype(bool);diag['h_subgroup_mean_error']=float(np.max(abs(CH[mask].mean(0)-P[mask].mean(0)))) if mask.any() else np.nan
    return results

def fit_super(W,X,Y,Z,seed=20261027,known_e=None,clip=10000.,n_estimators=500,return_folds=False,h=None):
    """X can be a matrix or mapping of nested matrices; shared nuisance fits.
    One random 5-fold partition independent of observations. All training
    pseudo-outcomes and reduced fits use only the complement of each fold.
    known_e can be a scalar, an n-vector, or None. No inner cross-fitting.
    """
    W=np.asarray(W,float);Y=np.asarray(Y,float);Z=np.asarray(Z,int);n=len(Z)
    if not np.isfinite(W).all() or not np.isfinite(Y).all():raise ValueError('finite complete-case W,Y required')
    xs=X if isinstance(X,dict) else {'X':np.asarray(X,float)}
    rng=np.random.default_rng(seed);fold=np.empty(n,int);fold[rng.permutation(n)]=np.arange(n)%5
    P=np.empty_like(Y);TT={k:np.empty_like(Y) for k in xs};ep=np.empty(n);mu0all=np.empty_like(Y);mu1all=np.empty_like(Y)
    eall=None if known_e is None else np.broadcast_to(known_e,(n,))
    for l in range(5):
        tr=fold!=l;te=~tr; wt=W[tr];zt=Z[tr];yt=Y[tr]
        m0=StumpBoost(n_estimators=n_estimators).fit(wt[zt==0],yt[zt==0]);m1=StumpBoost(n_estimators=n_estimators).fit(wt[zt==1],yt[zt==1])
        mu0=m0.predict(W[te]);mu1=m1.predict(W[te]);u0=m0.predict(wt);u1=m1.predict(wt)
        if known_e is None:
            pe=StumpBoost(n_estimators=n_estimators).fit(wt,zt);ee=np.clip(pe.predict(W[te]),.01,.99);et=np.clip(pe.predict(wt),.01,.99)
        else:ee=eall[te];et=eall[tr]
        ep[te]=ee;mu0all[te]=mu0;mu1all[te]=mu1
        P[te]=mu1-mu0+(Z[te]/ee)[:,None]*(Y[te]-mu1)-((1-Z[te])/(1-ee))[:,None]*(Y[te]-mu0)
        ptr=u1-u0+(zt/et)[:,None]*(yt-u1)-((1-zt)/(1-et))[:,None]*(yt-u0)
        for k,xx in xs.items():
            xx=np.asarray(xx,float)
            if xx.shape==W.shape and np.array_equal(xx,W):tt=mu1-mu0
            else:tt=StumpBoost(n_estimators=n_estimators).fit(xx[tr],ptr).predict(xx[te])
            TT[k][te]=np.clip(tt,-np.asarray(clip),np.asarray(clip))
    out={k:from_predictions(P,TT[k],h=(np.asarray(xx)[:,0]*np.asarray(xx)[:,1]>0) if h is None else h) for k,xx in xs.items()}
    for k in out:
        out[k]['nuisance']={'propensity':ep,'mu0':mu0all,'mu1':mu1all}
        out[k]['diagnostics'].update(propensity_min=float(ep.min()),propensity_max=float(ep.max()),propensity_truncated_fraction=float(np.mean((ep==.01)|(ep==.99))))
        if return_folds:out[k]['fold']=fold
    return out if isinstance(X,dict) else out['X']
