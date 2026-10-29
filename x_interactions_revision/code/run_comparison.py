#!/usr/bin/env python3
"""Compare W1=[X, nonconstant X-by-U interactions] on the original fixed populations and assignments."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, hashlib, json, platform, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from finite import fit_finite, finite_truth, NAMES, CRITICAL
from finite_sim import summarize
from population import population
from ding import fit_ding, ding_truth
from preflight_ding import exhaustive_check

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'specification.json').read_text())
RESULTS=ROOT/'results'
ARCHIVE=ROOT.parent/'targeted_revision/finite/results'
METHODS=list(SPEC['methods'])

def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def settings(s):
    X,U,irr,Y0,Y1,diag=population(s)
    ix=(X[:,1:,None]*U[:,None,:]).reshape(s['n'],-1)
    W0=np.c_[X,U,ix]
    Ws={'model_assisted':W0,'ma_x_interactions':np.c_[X,ix]}
    assert Ws['ma_x_interactions'].shape[1]==10
    assert np.linalg.matrix_rank(Ws['ma_x_interactions'])==10
    return X,U,irr,Y0,Y1,diag,Ws

def contrasts(e,se,truth,s):
    rows=[]
    pairs=[(m,'ols') for m in METHODS if m!='ols']
    pairs += [('ma_x_interactions','model_assisted'),('ding_plugin','model_assisted')]
    for a,b in pairs:
        for j,target in enumerate(NAMES):
            ea=e[a][:,j];eb=e[b][:,j];sa=se[a][:,j];sb=se[b][:,j]
            err_a=ea-truth[j];err_b=eb-truth[j]
            ok=np.isfinite(ea)&np.isfinite(eb)
            ci_a=np.isfinite(ea)&np.isfinite(sa)&(sa>0)
            ci_b=np.isfinite(eb)&np.isfinite(sb)&(sb>0)
            cov_a=ci_a&(abs(err_a)<=CRITICAL*sa)
            cov_b=ci_b&(abs(err_b)<=CRITICAL*sb)
            d=cov_a.astype(float)-cov_b.astype(float)
            row={**s,'method':a,'reference':b,'estimand':target,'reps':len(ea),
                 'coverage_difference':d.mean(),'coverage_difference_mcse':d.std(ddof=1)/np.sqrt(len(d)),
                 'paired_estimates':int(ok.sum()),'paired_intervals':int((ci_a&ci_b).sum())}
            erra=err_a[ok];errb=err_b[ok];count=len(erra)
            bd=erra-errb; ra=np.sqrt(np.mean(erra**2));rb=np.sqrt(np.mean(errb**2))
            infl=(erra**2-ra**2)/(2*ra)-(errb**2-rb**2)/(2*rb)
            row.update(bias_difference=bd.mean(),bias_difference_mcse=bd.std(ddof=1)/np.sqrt(count),
                       rmse_difference=ra-rb,rmse_difference_mcse=infl.std(ddof=1)/np.sqrt(count))
            good=ci_a&ci_b;aa=sa[good];bb=sb[good];nn=len(aa)
            ratio=aa.mean()/bb.mean();influence=(aa-ratio*bb)/bb.mean()
            less=(aa<bb).astype(float)
            row.update(width_ratio=ratio,width_ratio_mcse=influence.std(ddof=1)/np.sqrt(nn),
                       fraction_se_smaller=less.mean(),fraction_se_smaller_mcse=less.std(ddof=1)/np.sqrt(nn))
            rows.append(row)
    return rows

def run(s,reps,preflight=False):
    start=time.time();n=s['n'];n1=round(n*s['p'])
    X,U,irr,Y0,Y1,diag,Ws=settings(s)
    e={m:np.full((reps,8),np.nan) for m in METHODS}; se={m:np.full((reps,8),np.nan) for m in METHODS}
    ts={};gs={};ss={};bs={};bg={};tr={};fail=[];order=np.inf
    for m,W in Ws.items():
        t=finite_truth(X,Y0,Y1,n1/n,W)
        for key in (['ols',m] if m=='model_assisted' else [m]):
            v=t['ols' if key=='ols' else 'model_assisted']
            ts[key]=v['estimate'];ss[key]=v['sigma2'];gs[key]=v['gap'];bs[key]=v['beta_sigma'];bg[key]=v['beta_gap']
    dt=ding_truth(X,Y0,Y1,n1/n)
    ts['ding_plugin']=dt['estimate'];ss['ding_plugin']=dt['sigma2'];gs['ding_plugin']=dt['gap']
    bs['ding_plugin']=dt['beta_sigma'];bg['ding_plugin']=dt['beta_gap']
    for m in METHODS:
        np.testing.assert_allclose(ts[m],ts['ols'],rtol=2e-14,atol=2e-14)
        # Use exactly the same numeric truth for all four methods.
        ts[m]=ts['ols'].copy()
    assignment_hash=hashlib.sha256()
    for r in range(reps):
        rng=np.random.default_rng(np.random.SeedSequence([s['assignment_seed'],r]))
        Z=np.zeros(n,dtype=int);Z[rng.choice(n,n1,replace=False)]=1
        assert Z.sum()==n1
        assignment_hash.update(Z.astype(np.uint8).tobytes())
        Y=np.where(Z[:,None],Y1,Y0)
        for m,W in Ws.items():
            try:
                fitted=fit_finite(X,Y,Z,W)
                for key in (['ols',m] if m=='model_assisted' else [m]):
                    v=fitted['ols' if key=='ols' else 'model_assisted']
                    e[key][r]=v['estimate'];se[key][r]=v['se']
                order=min(order,float(np.linalg.eigvalsh(fitted['ols']['beta_cov']-fitted['model_assisted']['beta_cov'])[0]))
            except Exception as exc:
                fail.append({'replicate':r,'method':m,'error':repr(exc)})
        try:
            v=fit_ding(X,Y,Z)
            e['ding_plugin'][r]=v['estimate'];se['ding_plugin'][r]=v['se']
        except Exception as exc:
            fail.append({'replicate':r,'method':'ding_plugin','error':repr(exc)})
    archived={}
    prior=ARCHIVE/(s['scenario']+'_raw.npz')
    if prior.exists():
        old=np.load(prior)
        for m in ['ols','model_assisted','ding_plugin']:
            for name,arr in [('estimate',e[m]),('se',se[m])]:
                oldarr=old[f'large__{m}__{name}'][:reps]
                diff=float(np.nanmax(abs(arr-oldarr)))
                np.testing.assert_allclose(arr,oldarr,rtol=2e-13,atol=2e-13,equal_nan=True)
                archived[f'{m}_{name}_max_abs_difference']=diff
    else:
        archived['status']='Archive not available; generation still self-contained.'
    manifest={**s,**diag,'reps':reps,'attempted_assignments':reps,
        'attempted_method_fits':reps*len(METHODS),'failures':fail,'seconds':time.time()-start,
        'W_columns':{m:W.shape[1] for m,W in Ws.items()},
        'W_population_rank':{m:int(np.linalg.matrix_rank(W)) for m,W in Ws.items()},
        'population_hashes':{key:digest(val) for key,val in [('X',X),('U',U),('irrelevant',irr),('Y0',Y0),('Y1',Y1)]},
        'assignment_uint8_concatenation_sha256':assignment_hash.hexdigest(),
        'archive_reproduction':archived,'min_coefficient_covariance_ordering_eigenvalue':order,
        'estimate_success':{m:int(np.isfinite(e[m]).sum()) for m in METHODS},
        'interval_success':{m:int((np.isfinite(e[m])&np.isfinite(se[m])&(se[m]>0)).sum()) for m in METHODS}}
    if preflight:
        return manifest
    payload={}
    rows=[]
    for m in METHODS:
        rows+=summarize(e[m],se[m],ts[m],ss[m],gs[m],n,{**s,**diag},m,'large',reps,se['ols'])
        for key,value in [('estimate',e[m]),('se',se[m]),('truth',ts[m]),('sigma2',ss[m]),('gap',gs[m]),('beta_sigma',bs[m]),('beta_gap',bg[m])]:
            payload[f'large__{m}__{key}']=value
    np.savez_compressed(RESULTS/(s['scenario']+'_raw.npz'),**payload)
    np.savez_compressed(RESULTS/(s['scenario']+'_population.npz'),X=X,U=U,irrelevant=irr,Y0=Y0,Y1=Y1)
    pd.DataFrame(rows).to_csv(RESULTS/(s['scenario']+'_summary.csv'),index=False)
    pd.DataFrame(contrasts(e,se,ts['ols'],s)).to_csv(RESULTS/(s['scenario']+'_paired.csv'),index=False)
    (RESULTS/(s['scenario']+'_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.preflight:
        ding_check=exhaustive_check()
        (RESULTS/'ding_exhaustive_check.json').write_text(json.dumps(ding_check,indent=2)+'\n')
        checks=[run(s,10,True) for s in SPEC['scenarios']]
        (RESULTS/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'status':'passed','cells':len(checks),'reps_each':10,'failures':sum(len(x['failures']) for x in checks),'ding_exhaustive':ding_check}),flush=True)
        return
    manifest={'specification_sha256':hashlib.sha256((ROOT/'specification.json').read_bytes()).hexdigest(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))},
        'reps':SPEC['reps'],'workers':SPEC['workers'],'python':platform.python_version(),'numpy':np.__version__,
        'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    (ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    completed=[]
    with ProcessPoolExecutor(SPEC['workers']) as pool:
        futures=[pool.submit(run,s,SPEC['reps']) for s in SPEC['scenarios']]
        for f in as_completed(futures):
            r=f.result();completed.append(r)
            print(json.dumps({'n':r['n'],'seconds':round(r['seconds'],3),'failures':len(r['failures'])}),flush=True)
    for suffix,target in [('summary','summary'),('paired','paired_contrasts')]:
        tables=[pd.read_csv(RESULTS/(s['scenario']+f'_{suffix}.csv')) for s in SPEC['scenarios']]
        pd.concat(tables,ignore_index=True).to_csv(RESULTS/(target+'.csv'),index=False)
    manifest.update(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),completed_scenarios=len(completed),
        attempted_assignments=sum(r['reps'] for r in completed),attempted_method_fits=sum(r['attempted_method_fits'] for r in completed),
        recorded_scalar_estimates=sum(sum(r['estimate_success'].values()) for r in completed),
        failures=sum(len(r['failures']) for r in completed))
    (ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
