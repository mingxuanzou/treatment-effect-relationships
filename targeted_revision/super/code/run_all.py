#!/usr/bin/env python3
"""Reproduce all five plotted superpopulation estimators from frozen seeds.

Results are written in a separate directory; archived inputs are never modified.
Use --preflight for three datasets at each n or omit it for all 1000 per n.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,hashlib,sys,gzip
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.stats import skew
from super_v4 import dgp,truth,fit_super_v2,efficiency,NAMES
from super import from_predictions

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT.parents[1]
SPEC=json.loads((ROOT/'specification.json').read_text())
ARCHIVE=PACKAGE/'work/package/experiment/revision4/results'
METHODS=['direct','calibrated','naive','oracle','x_only']
SETTINGS=['rho','s1','s2','noise_corr','common','design']

def one(job):
    cell,rep=job
    seed=cell['data_seed_base']+rep; split=cell['split_seed_base']+rep
    base={k:v for k,v in cell.items() if k not in ('data_seed_base','split_seed_base')}
    base.update(rep=rep,seed=seed,split_seed=split,learner='cubic_ridge',nested='large')
    W,Y,Z,c=dgp(cell['n'],seed,**{k:cell[k] for k in SETTINGS})
    target=truth(cell['rho'],cell['s1'],cell['s2'],'large')
    fit=fit_super_v2(W,W[:,:3],Y,Z,seed=split,learner='cubic_ridge',clip=12.)
    fit['x_only']=fit_super_v2(W[:,:3],W[:,:3],Y,Z,seed=split,learner='cubic_ridge',clip=12.)['direct']
    P=c['fulltau']+(Z/c['e'])[:,None]*(Y-c['mu1'])-((1-Z)/(1-c['e']))[:,None]*(Y-c['mu0'])
    fit['oracle']=from_predictions(P,c['tau']['large'])['direct']
    rows=[]
    for method in METHODS:
        res=fit[method]
        for j,name in enumerate(NAMES):
            est,se,lo,hi=[res[k][j] for k in ('estimate','se','lower','upper')]
            valid=bool(np.isfinite([est,se,lo,hi]).all() and se>0)
            rows.append(dict(base,method=method,estimand=name,truth=target[j],estimate=est,se=se,lower=lo,upper=hi,valid=valid,covered=bool(valid and lo<=target[j]<=hi),fit_success=True))
    return rows

def summary(raw):
    settings={k:SPEC["cells"][0][k] for k in SETTINGS}
    sigma_values=dict(zip(NAMES,efficiency(settings,"large")))
    rows=[]
    keys=['cell_id','kind','n','design','rho','s1','s2','weak','strength','learner','nested','method','estimand']
    for key,g in raw.groupby(keys,sort=False):
        row=dict(zip(keys,key));p=g[np.isfinite(g.estimate)];v=g[g.valid];R=len(g);target=g.truth.iloc[0]
        errors=p.estimate-target;sd=p.estimate.std(ddof=1);rmse=np.sqrt(np.mean(errors**2));cov=g.covered.mean()
        sigma=float(sigma_values[row['estimand']])
        row.update(truth=target,attempted=R,returned=R,fit_failed=0,point_defined=len(p),valid=len(v),undefined=R-len(v),bias=errors.mean(),bias_mcse=sd/np.sqrt(len(p)),sd=sd,rmse=rmse,rmse_mcse=(errors**2).std(ddof=1)/(2*rmse*np.sqrt(len(p))) if rmse>0 else 0.,mean_se=v.se.mean(),se_sd=v.se.mean()/sd,coverage=cov,coverage_mcse=np.sqrt(cov*(1-cov)/R),coverage_conditional=v.covered.mean(),width=(v.upper-v.lower).mean(),sigma_eif=sigma,sqrt_n_sd_over_sigma=np.sqrt(row['n'])*sd/sigma,root_n_bias=np.sqrt(row['n'])*errors.mean(),standardized_bias=np.sqrt(row['n'])*errors.mean()/sigma,relative_bias=errors.mean()/target if target else np.nan,estimate_q025=p.estimate.quantile(.025),estimate_median=p.estimate.median(),estimate_q975=p.estimate.quantile(.975),interval_outside_unit=((v.lower< -1)|(v.upper>1)).mean() if row['estimand']=='rho' else np.nan,estimate_skew=skew(p.estimate) if len(p)>3 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)

def compare(raw,cell,reps):
    a=pd.read_csv(ARCHIVE/(cell['cell_id']+'_raw.csv.gz'))
    a=a[(a.nested=='large')&a.method.isin(METHODS)]
    b=pd.read_csv(ROOT/'results'/(cell['cell_id']+'_raw.csv.gz'))
    old=pd.concat([a,b],ignore_index=True);old=old[old.rep<reps]
    keys=['rep','method','estimand']
    merged=raw.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(merged)==reps*len(METHODS)*len(NAMES)
    differences={}
    for col in ['estimate','se','lower','upper','truth']:
        np.testing.assert_allclose(merged[col+'_new'],merged[col+'_old'],rtol=1e-9,atol=1e-10,equal_nan=True)
        differences[col]=float(np.nanmax(abs(merged[col+'_new']-merged[col+'_old'])))
    for col in ['valid','covered']:
        assert np.array_equal(merged[col+'_new'],merged[col+'_old'])
    return dict(n=cell['n'],replications=reps,records=len(merged),max_abs_differences=differences,coverage_indicators_identical=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--workers',type=int,default=SPEC['workers']);p.add_argument('--output',type=Path);p.add_argument('--refresh-plot-data',action='store_true');p.add_argument('--verify-archive',action='store_true');p.add_argument('--summarize-only',action='store_true');a=p.parse_args()
    a.output=a.output or ROOT/('preflight_results' if a.preflight else 'reproduced_results')
    a.output.mkdir(parents=True,exist_ok=True);reps=3 if a.preflight else SPEC['reps'];start=time.time();checks=[];frames=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for cell in SPEC['cells']:
            if a.summarize_only:
                raw=pd.read_csv(a.output/(cell['cell_id']+'_raw.csv.gz'))
            else:
                rows=[]
                for batch in pool.map(one,[(cell,r) for r in range(reps)],chunksize=5): rows.extend(batch)
                raw=pd.DataFrame(rows)
            frames.append(raw)
            checks.append(compare(raw,cell,reps) if a.verify_archive else dict(n=cell['n'],replications=reps,records=len(raw)))
            if not a.summarize_only:
                payload=gzip.compress(raw.to_csv(index=False).encode(),mtime=0)
                destination=a.output/(cell['cell_id']+'_raw.csv.gz')
                temporary=destination.with_suffix('.tmp')
                temporary.write_bytes(payload);temporary.replace(destination)
                assert len(pd.read_csv(destination))==len(raw)
            print(json.dumps(checks[-1]),flush=True)
    raw=pd.concat(frames,ignore_index=True);s=summary(raw);s.to_csv(a.output/'summary.csv',index=False)
    if a.verify_archive and not a.preflight:
        plotted=pd.read_csv(PACKAGE/'simulation/source_data/super_confirm_summary.csv')
        keys=['cell_id','n','method','estimand'];old=plotted[plotted.nested.eq('large')&plotted.design.eq('observational')&plotted.method.isin(METHODS)]
        q=s.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        for metric in ['bias','rmse','coverage']:
            np.testing.assert_allclose(q[metric+'_new'],q[metric+'_old'],rtol=1e-9,atol=1e-10,equal_nan=True)
    if a.refresh_plot_data and not a.preflight:s.to_csv(PACKAGE/'simulation/source_data/super_confirm_summary.csv',index=False)
    report=dict(status='passed',preflight=a.preflight,summarize_only=a.summarize_only,attempted_datasets=len(SPEC['cells'])*reps,scalar_records=len(raw),checks=checks,seconds=time.time()-start,python=sys.version,numpy=np.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code').glob('*.py')},summary_metrics_match_archived_plot_data=(a.verify_archive and not a.preflight))
    (a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
