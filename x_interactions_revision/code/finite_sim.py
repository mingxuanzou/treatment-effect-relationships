#!/usr/bin/env python3
"""Fixed-schedule rerandomization simulations. Run --reps 2000 --workers 2."""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[v]='1'
import argparse,hashlib,json,time,sys,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import skew
from finite import fit_finite,finite_truth,moments,unbiased_coefficients,NAMES
ROOT=Path(__file__).resolve().parents[1]

def seed_for(text): return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4],'little')

def population(n,rho=.5,strength=1.,weak='none',kappa=0.,seed=421061):
    rng=np.random.default_rng(seed)
    raw=rng.uniform(-np.sqrt(3),np.sqrt(3),(n,7))
    # Recursive orthogonalization preserves nested projection spaces.
    Q=[]
    for j in range(3):
        q=raw[:,j]-raw[:,j].mean()
        for old in Q:q=q-old*(old@q)/(old@old)
        Q.append(q/np.std(q,ddof=1))
    x=np.array(Q).T;X=np.c_[np.ones(n),x];u=raw[:,3];irr=raw[:,4:]
    a=np.array([np.sqrt(.4),0,np.sqrt(.6)])
    orth=.8*np.array([-np.sqrt(.6),0,np.sqrt(.4)])+.6*np.array([0,1,0])
    b=rho*a+np.sqrt(1-rho**2)*orth
    s1,s2=1.,.7
    if weak=='both':s1,s2=strength,.7*strength
    if weak=='second':s2=strength
    eps=rng.normal(size=(n,4));eps[:,1]=.4*eps[:,0]+np.sqrt(.84)*eps[:,1]
    h=eps[:,2:];h-=X@np.linalg.solve(X.T@X,X.T@h);h/=h.std(axis=0,ddof=1)
    tau=np.c_[1+s1*(x@a),-.5+s2*(x@b)]+kappa*h
    base=np.c_[1.5*u+.6*(x[:,0]**2-1)+.4*np.sin(x[:,1]),1.2*u+.5*(x[:,1]**2-1)-.3*np.cos(x[:,0])]
    Y0=base+eps[:,:2];Y1=Y0+tau
    return X,u,irr,Y0,Y1

def scenario_grid():
    S=[]
    def add(family,n,p=.5,kappa=0.,rho=.5,strength=1.,weak='none',extra=False):
        S.append(dict(family=family,n=n,p=p,kappa=kappa,rho=rho,strength=strength,weak=weak,extra=extra))
    for n in (200,500,1000,2000,5000):
        for p in (.3,.5):
            for kappa in (0.,.5,1.,2.): add('F1',n,p,kappa,extra=(n==2000 and p==.5 and kappa==1.))
    for n in (1000,5000):
        for rho in (.8,.9,.95,.99): add('B1',n,rho=rho)
    for n in (1000,5000):
        for weak in ('both','second'):
            for s in (.05,.1,.2,.5):add('B2',n,strength=s,weak=weak)
    for idx,s in enumerate(S):s['scenario']=f"fp_{idx:03d}";s['population_seed']=seed_for(f"FP-population-v1-n{s['n']}");s['assignment_seed']=seed_for(f"FP-assign-v1-n{s['n']}-p{s['p']}")
    return S

def summarize(e,se,truth,sigma,gap,n,scenario,method,xset,reps,base_se=None):
    rows=[]
    for j,name in enumerate(NAMES):
        ok=np.isfinite(e[:,j]);oki=ok&np.isfinite(se[:,j])&(se[:,j]>0)
        err=e[ok,j]-truth[j];cov=oki&(abs(e[:,j]-truth[j])<=1.959963984540054*se[:,j])
        R=ok.sum();sd=np.std(e[ok,j],ddof=1); mse=np.mean(err**2);rmse=np.sqrt(mse);cover=np.mean(cov)
        row={**scenario,'xset':xset,'method':method,'estimand':name,'reps':reps,'estimate_success':int(R),'interval_success':int(oki.sum()),'truth':truth[j],
          'bias':np.mean(err),'bias_mcse':sd/np.sqrt(R),'sd':sd,'sqrt_n_sd':np.sqrt(n)*sd,'rmse':rmse,
          'rmse_mcse':np.std(err**2,ddof=1)/np.sqrt(R)/(2*rmse),'mean_se':np.mean(se[oki,j]),'se_sd_ratio':np.mean(se[oki,j])/sd,
          'coverage':cover,'coverage_mcse':np.sqrt(cover*(1-cover)/reps),'coverage_available':np.mean(cov[oki]),
          'mean_width':2*1.959963984540054*np.mean(se[oki,j]),'sigma2':sigma[j],'gap':gap[j],
          'n_variance_sigma2_ratio':n*sd**2/sigma[j],'variance_target_ratio':n*np.mean(se[oki,j]**2)/(sigma[j]+gap[j]),
          'theory_se_sd_ratio':np.sqrt(1+gap[j]/sigma[j]),'relative_bias':np.mean(err)/truth[j] if truth[j]!=0 else np.nan,
          'interval_beyond_domain':np.mean(oki&((e[:,j]+1.959963984540054*se[:,j]>1)|(e[:,j]-1.959963984540054*se[:,j]<-1))) if name=='rho' else np.nan,
          'estimate_q025':np.quantile(e[ok,j],.025),'estimate_q50':np.quantile(e[ok,j],.5),'estimate_q975':np.quantile(e[ok,j],.975),'skewness':skew(e[ok,j],bias=False)}
        if base_se is not None:
            good=oki&np.isfinite(base_se[:,j])&(base_se[:,j]>0)
            row['width_ratio_vs_ols']=np.mean(se[good,j])/np.mean(base_se[good,j]);row['fraction_se_smaller']=np.mean(se[good,j]<base_se[good,j])
            basecov=np.isfinite(e[:,j]) # actual paired coverage contrasts computed separately below
        rows.append(row)
    return rows

def run_one(args):
    s,reps,outdir=args;t0=time.time();outdir=Path(outdir)
    Xfull,u,irr,Y0,Y1=population(s['n'],s['rho'],s['strength'],s['weak'],s['kappa'],s['population_seed'])
    n=s['n'];n1=round(n*s['p']); p=n1/n
    xsets={'small':Xfull[:,:3],'large':Xfull}
    estimates={};ses={};truths={};sigmas={};gaps={};Ws={};failures=[];ranks={}
    for xn,X in xsets.items():
        Ws[xn]={'model_assisted':np.c_[X,u,X[:,1:]*u[:,None]]}
        if s['extra']:Ws[xn].update({'ma_X':X,'ma_main':np.c_[X,u],'ma_irrelevant':np.c_[X,irr]})
        for method,W in Ws[xn].items():
            tr=finite_truth(X,Y0,Y1,p,W)
            for key in (['ols',method] if method=='model_assisted' else [method]):
                val=tr['ols' if key=='ols' else 'model_assisted']; k=f'{xn}/{key}'
                truths[k]=val['estimate'];sigmas[k]=val['sigma2'];gaps[k]=val['gap'];estimates[k]=np.full((reps,8),np.nan);ses[k]=np.full((reps,8),np.nan)
    shifts=np.full((reps,2,3,2,8),np.nan) if s['extra'] else None
    coeff_errors=np.full((reps,2,3,2),np.nan) if s['extra'] else None
    min_order=np.inf
    for r in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([s['assignment_seed'],r]))
        Z=np.zeros(n,dtype=int);Z[rng.choice(n,n1,replace=False)]=1;Y=np.where(Z[:,None],Y1,Y0)
        for xi,(xn,X) in enumerate(xsets.items()):
            for method,W in Ws[xn].items():
                try:
                    f=fit_finite(X,Y,Z,W)
                    for key in (['ols',method] if method=='model_assisted' else [method]):
                        res=f['ols' if key=='ols' else 'model_assisted'];k=f'{xn}/{key}';estimates[k][r]=res['estimate'];ses[k][r]=res['se']
                    min_order=min(min_order,float(np.linalg.eigvalsh(f['ols']['beta_cov']-f['model_assisted']['beta_cov'])[0]))
                except Exception as ex:failures.append(dict(replicate=r,xset=xn,method=method,error=repr(ex)))
            if s['extra']:
                bt=np.linalg.solve(X.T@X,X.T@(Y1-Y0))
                for ci,c in enumerate((0.,2.,5.)):
                    for mi,method in enumerate(('ols','unbiased')):
                        if method=='ols':b=fit_finite(X,Y+c,Z)['ols']['beta']
                        else:b=unbiased_coefficients(X,Y+c,Z)
                        shifts[r,xi,ci,mi]=moments(X,b)[0];coeff_errors[r,xi,ci,mi]=np.sum((b-bt)**2)
    rows=[]
    for k in estimates:
        xn,method=k.split('/');rows+=summarize(estimates[k],ses[k],truths[k],sigmas[k],gaps[k],n,s,method,xn,reps,ses[f'{xn}/ols'])
    payload={}
    for k in estimates:
        safe=k.replace('/','__');payload[safe+'__estimate']=estimates[k];payload[safe+'__se']=ses[k];payload[safe+'__truth']=truths[k];payload[safe+'__sigma2']=sigmas[k];payload[safe+'__gap']=gaps[k]
    if shifts is not None:payload['shift_estimates']=shifts;payload['shift_coefficient_squared_errors']=coeff_errors
    np.savez_compressed(outdir/(s['scenario']+'_raw.npz'),**payload)
    pd.DataFrame(rows).to_csv(outdir/(s['scenario']+'_summary.csv'),index=False)
    manifest={**s,'reps':reps,'attempted_assignment_replicates':reps,'failures':failures,'min_covariance_ordering_eigenvalue':min_order,'seconds':time.time()-t0}
    (outdir/(s['scenario']+'_manifest.json')).write_text(json.dumps(manifest,indent=2))
    return manifest

def aggregate(outdir):
    outdir=Path(outdir);D=pd.concat([pd.read_csv(p) for p in sorted(outdir.glob('fp_*_summary.csv'))],ignore_index=True)
    D.to_csv(ROOT/'results'/'finite_summary.csv',index=False)
    # Paired outcome-shift experiment (unbiasedness is a coefficient property).
    S=scenario_grid();s=next(s for s in S if s['extra']);path=outdir/(s['scenario']+'_raw.npz')
    if path.exists():
        a=np.load(path);rows=[]
        for xi,x in enumerate(('small','large')):
            tr=a[x+'__ols__truth'];e=a['shift_estimates'][:,xi];ce=a['shift_coefficient_squared_errors'][:,xi]
            for ci,c in enumerate((0,2,5)):
                for mi,method in enumerate(('ols','unbiased')):
                    for j,nm in enumerate(NAMES):
                        err=e[:,ci,mi,j]-tr[j];oerr=e[:,ci,0,j]-tr[j]
                        rows.append(dict(xset=x,c=c,method=method,estimand=nm,reps=len(e),truth=tr[j],bias=err.mean(),rmse=np.sqrt(np.mean(err**2)),rmse_vs_ols=np.sqrt(np.mean(err**2)/np.mean(oerr**2)),coefficient_rmse=np.sqrt(ce[:,ci,mi].mean()),coefficient_rmse_vs_ols=np.sqrt(ce[:,ci,mi].mean()/ce[:,ci,0].mean())))
        pd.DataFrame(rows).to_csv(ROOT/'results'/'finite_shift_summary.csv',index=False)
    return D

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--reps',type=int,default=2000);pa.add_argument('--workers',type=int,default=2);pa.add_argument('--pilot',action='store_true');pa.add_argument('--aggregate',action='store_true');args=pa.parse_args()
    out=ROOT/'results'/('finite_pilot' if args.pilot else 'finite_raw');out.mkdir(exist_ok=True)
    if args.aggregate:aggregate(out);return
    S=scenario_grid();S=S[:1] if args.pilot else S
    manifest=dict(specification='finite-v1',reps=args.reps,workers=args.workers,scenarios=S,python=sys.version,numpy=np.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('finite.py')]})
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    tasks=[(s,args.reps,str(out)) for s in S]
    with ProcessPoolExecutor(args.workers) as pool:
        futures=[pool.submit(run_one,t) for t in tasks]
        for f in as_completed(futures):
            r=f.result();print(r['scenario'],round(r['seconds'],2),'seconds',len(r['failures']),'failures',flush=True)
    if not args.pilot:aggregate(out)
if __name__=='__main__':main()
