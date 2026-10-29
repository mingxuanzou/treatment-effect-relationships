#!/usr/bin/env python3
"""Check saved simulation results and figure inputs without historical archives."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import norm
ROOT=Path(__file__).resolve().parent
if not (ROOT/'targeted_revision/super/reproduced_results/v4c_00_raw.csv.gz').exists() or not (ROOT/'x_interactions_revision/results/fw_n200_raw.npz').exists():
    raise SystemExit('Run python run_replication.py simulations first to create the raw replication records.')
NAMES=['ATE1','ATE2','V11','V22','V12','rho','theta','alpha']
CRIT=norm.ppf(.975)

def targets(X,Y0,Y1):
    B=np.linalg.lstsq(X,Y1-Y0,rcond=None)[0]
    fitted=X@B;a=fitted.mean(0);v=np.cov(fitted,rowvar=False,ddof=1)
    slope=v[0,1]/v[0,0]
    return np.r_[a,v[0,0],v[1,1],v[0,1],v[0,1]/np.sqrt(v[0,0]*v[1,1]),slope,a[1]-slope*a[0]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'simulation_verification.json');a=p.parse_args()
    finite=ROOT/'x_interactions_revision';fspec=json.loads((finite/'specification.json').read_text());s=pd.read_csv(finite/'results/summary.csv');fc=[]
    for cell in fspec['scenarios']:
        r=np.load(finite/'results'/(cell['scenario']+'_raw.npz'));pop=np.load(finite/'results'/(cell['scenario']+'_population.npz'))
        truth=targets(pop['X'],pop['Y0'],pop['Y1'])
        for method in fspec['methods']:
            prefix='large__'+method+'__';e=r[prefix+'estimate'];se=r[prefix+'se'];tr=r[prefix+'truth']
            np.testing.assert_allclose(tr,truth,rtol=1e-12,atol=1e-12)
            assert len(e)==fspec['reps']
            for j,name in enumerate(NAMES):
                ok=np.isfinite(e[:,j]);valid=ok&np.isfinite(se[:,j])&(se[:,j]>0);err=e[:,j]-truth[j]
                cov=valid&(abs(err)<=CRIT*se[:,j]);row=s[(s.n==cell['n'])&(s.method==method)&(s.estimand==name)].iloc[0]
                np.testing.assert_allclose([err[ok].mean(),np.sqrt(np.mean(err[ok]**2)),cov.mean()],[row.bias,row.rmse,row.coverage],rtol=1e-11,atol=1e-12)
        fc.append(dict(n=cell['n'],assignments=fspec['reps'],methods=len(fspec['methods'])))
    superroot=ROOT/'targeted_revision/super';sp=json.loads((superroot/'specification.json').read_text());ss=pd.read_csv(superroot/'reproduced_results/summary.csv');sc=[]
    for cell in sp['cells']:
        raw=pd.read_csv(superroot/'reproduced_results'/(cell['cell_id']+'_raw.csv.gz'))
        for (method,name),g in raw.groupby(['method','estimand']):
            assert len(g)==sp['reps']
            valid=np.isfinite(g[['estimate','se','lower','upper']]).all(axis=1)&(g.se>0)
            cov=valid&(g.lower<=g.truth)&(g.truth<=g.upper)
            assert np.array_equal(valid,g.valid);assert np.array_equal(cov,g.covered)
            ok=np.isfinite(g.estimate);err=(g.estimate-g.truth)[ok]
            row=ss[(ss.cell_id==cell['cell_id'])&(ss.method==method)&(ss.estimand==name)].iloc[0]
            np.testing.assert_allclose([err.mean(),np.sqrt(np.mean(err**2)),cov.mean()],[row.bias,row.rmse,row.coverage],rtol=1e-11,atol=1e-12)
        sc.append(dict(n=cell['n'],datasets=sp['reps'],methods=raw.method.nunique()))
    for name,current in [('super_confirm_summary.csv',ss),('finite_w_summary.csv',s)]:
        plot=pd.read_csv(ROOT/'simulation/source_data'/name);keys=['n','method','estimand'];plot=plot[plot.method.isin(current.method)]
        if 'nested' in plot:plot=plot[plot.nested.eq('large')&plot.design.eq('observational')]
        merged=plot.merge(current,on=keys,suffixes=('_plot','_computed'),validate='one_to_one')
        for metric in ['bias','rmse','coverage']:np.testing.assert_allclose(merged[metric+'_plot'],merged[metric+'_computed'],rtol=1e-11,atol=1e-12)
    result=dict(status='passed',finite_population=fc,superpopulation=sc,coverage_denominator='all attempted assignments or datasets',bias_rmse_denominator='finite point estimates',truths_independently_recomputed=True,figure_inputs_match_results=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
