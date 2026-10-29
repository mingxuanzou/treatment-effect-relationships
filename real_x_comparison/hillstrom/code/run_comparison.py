"""Four frozen Hillstrom X choices; observed-data OLS and model assistance only.

Run from any directory: python real_x_comparison/hillstrom/code/run_comparison.py
All paths are local to this self-contained analysis folder.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import gzip
import hashlib
import io
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from finite import fit_finite, NAMES, CRITICAL

ROOT=Path(__file__).resolve().parents[1]
SPEC=json.loads((ROOT/'spec.json').read_text())
OUT=ROOT/'results'


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def serial(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,(np.integer,np.floating,np.bool_)):return v.item()
    if isinstance(v,dict):return {k:serial(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)):return [serial(a) for a in v]
    return v


def load_data():
    rawbytes=gzip.decompress((ROOT/'input/hillstrom_raw.csv.gz').read_bytes())
    assert hashlib.sha256(rawbytes).hexdigest()==SPEC['source_csv_sha256']
    raw=pd.read_csv(io.BytesIO(rawbytes))
    d=raw.loc[raw.segment.isin(['Mens E-Mail','No E-Mail'])].copy()
    required=['recency','history','mens','womens','newbie','zip_code','channel','visit','spend']
    assert not d[required].isna().any().any()
    Z=(d.segment=='Mens E-Mail').to_numpy(dtype=int)
    Y=d[['visit','spend']].to_numpy(float)
    base=pd.DataFrame({'recency':d.recency,'log1p_history':np.log1p(d.history),
                       'mens':d.mens,'womens':d.womens},index=d.index,dtype=float)
    base=(base-base.mean())/base.std(ddof=0)
    xlarge=np.column_stack([np.ones(len(d)),base.to_numpy()])
    xsmall=xlarge[:,:3]
    extra=pd.concat([d[['newbie']].astype(float),
        pd.get_dummies(d.zip_code,prefix='zip',drop_first=True,dtype=float),
        pd.get_dummies(d.channel,prefix='channel',drop_first=True,dtype=float)],axis=1)
    extra['recency_by_newbie']=base.recency*d.newbie
    extra['log1p_history_by_newbie']=base.log1p_history*d.newbie
    extra=(extra-extra.mean())/extra.std(ddof=0)
    W=np.column_stack([xlarge,extra.to_numpy()])
    Wnames=['intercept',*base.columns,*extra.columns]
    assert Wnames==SPEC['W_names']
    Xs={'small':xsmall,'large':xlarge,
        'history_newbie':np.column_stack([xsmall,W[:,Wnames.index('newbie')]]),
        'combined':np.column_stack([xlarge,W[:,Wnames.index('newbie')]])}
    assert list(Xs)==list(SPEC['X_choices'])
    for label,X in Xs.items():
        expected=W[:,[Wnames.index(name) for name in SPEC['X_choices'][label]['names']]]
        np.testing.assert_array_equal(X,expected)
        assert np.linalg.matrix_rank(X)==X.shape[1]
        for z in (0,1):assert np.linalg.matrix_rank(X[Z==z])==X.shape[1]
    for z in (0,1):assert np.linalg.matrix_rank(W[Z==z])==W.shape[1]
    assert (len(d),int(Z.sum()),int((1-Z).sum()))==(SPEC['n'],SPEC['n1'],SPEC['n0'])
    current={'X_small':xsmall,'X_large':xlarge,'W':W,'Y':Y,'Z':Z,'original_row':d.index.to_numpy()}
    if (ROOT/'input/archive_arrays.npz').exists():
        original=np.load(ROOT/'input/archive_arrays.npz')
        current={'X_small':xsmall,'X_large':xlarge,'W':W,'Y':Y,'Z':Z,'original_row':d.index.to_numpy()}
        for name,arr in current.items():np.testing.assert_array_equal(arr,original[name])
    groups=d.groupby(['mens','womens'],observed=True).size()
    info={'n':len(d),'n1':int(Z.sum()),'n0':int((1-Z).sum()),
          'historical_array_checks_run':(ROOT/'input/archive_arrays.npz').exists(),'W_columns':Wnames,'W_rank':int(np.linalg.matrix_rank(W)),
          'X_ranks':{label:int(np.linalg.matrix_rank(X)) for label,X in Xs.items()},
          'input_array_hashes':{key:digest(value) for key,value in current.items()},
          'X_hashes':{label:digest(X) for label,X in Xs.items()},
          'category_group_counts':{f'mens={int(m)},womens={int(w)}':int(count) for (m,w),count in groups.items()}}
    return Xs,W,Y,Z,d.index.to_numpy(),info


def main():
    OUT.mkdir(exist_ok=True)
    started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    Xs,W,Y,Z,row,info=load_data()
    rows=[];full={};failures=[];diagnostics=[]
    for label,X in Xs.items():
        try:full[label]=fit_finite(X,Y,Z,W)
        except Exception as exc:
            failures.append({'X':label,'error':repr(exc)})
            full[label]={method:{key:np.full(8,np.nan) for key in ('estimate','se','lower','upper')} for method in SPEC['methods']}
            for v in full[label].values():v['interval_success']=np.zeros(8,dtype=bool)
        for method,v in full[label].items():
            for j,name in enumerate(NAMES):
                rows.append({'dataset':'Hillstrom','framework':'finite','X':label,
                    'X_label':SPEC['X_choices'][label]['label'],'method':method,'estimand':name,
                    'estimate':float(v['estimate'][j]),'se':float(v['se'][j]),
                    'lower':float(v['lower'][j]),'upper':float(v['upper'][j]),
                    'n':len(Y),'n1':int(Z.sum()),'defined':bool(v['interval_success'][j])})
            diagnostics.append({'X':label,'method':method,'X_rank':int(np.linalg.matrix_rank(X)),
                'W_rank':int(np.linalg.matrix_rank(W)),
                'V11_over_SE':float(v['estimate'][2]/v['se'][2]),
                'V22_over_SE':float(v['estimate'][3]/v['se'][3]),
                'rho':float(v['estimate'][5]),'rho_interval_outside':bool(v['lower'][5]<-1 or v['upper'][5]>1),
                'defined_intervals':int(v['interval_success'].sum())})
    archive_checks={}
    old=json.loads((ROOT/'input/archive_full.json').read_text()) if (ROOT/'input/archive_full.json').exists() else None
    if old is not None:
        for label in ('small','large'):
            for method in SPEC['methods']:
                for key in ('estimate','se','lower','upper','beta','beta_cov','gradient'):
                    current=np.asarray(full[label][method][key]);prior=np.asarray(old[label][method][key])
                    np.testing.assert_allclose(current,prior,rtol=2e-12,atol=2e-12,equal_nan=True)
                    archive_checks[f'{label}/{method}/{key}']=float(np.nanmax(abs(current-prior)))
    previous_path=ROOT/'archive/previous_four_X/results/full.json'
    previous_checks={}
    if previous_path.exists():
        previous=json.loads(previous_path.read_text())
        for label in ('small','large','history_newbie'):
            for method in SPEC['methods']:
                for key in ('estimate','se','lower','upper','beta','beta_cov','gradient'):
                    current=np.asarray(full[label][method][key]);prior=np.asarray(previous[label][method][key])
                    np.testing.assert_array_equal(current,prior)
                    previous_checks[f'{label}/{method}/{key}']=float(np.nanmax(abs(current-prior)))
    estimates=pd.DataFrame(rows)
    relationships=estimates.loc[estimates.estimand.isin(NAMES[2:])].copy()
    estimates.to_csv(OUT/'estimates.csv',index=False)
    relationships.to_csv(OUT/'relationship_estimates.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'diagnostics.csv',index=False)
    (OUT/'full.json').write_text(json.dumps(serial(full),indent=2)+'\n')
    (OUT/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    np.savez_compressed(OUT/'analysis_arrays.npz',**{f'X_{k}':v for k,v in Xs.items()},W=W,Y=Y,Z=Z,original_row=row)
    wide=[]
    for label in Xs:
        for method in SPEC['methods']:
            d={'X':label,'X_label':SPEC['X_choices'][label]['label'],'method':method}
            for name in NAMES[2:]:
                r=relationships.loc[(relationships.X==label)&(relationships.method==method)&(relationships.estimand==name)].iloc[0]
                d[name]=f'{r.estimate:.5g} ({r.lower:.5g}, {r.upper:.5g})'
            wide.append(d)
    pd.DataFrame(wide).to_csv(OUT/'relationship_table.csv',index=False)
    manifest={**info,'started_utc':started,'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'specification_sha256':hashlib.sha256((ROOT/'spec.json').read_bytes()).hexdigest(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'code').glob('*.py'))},
        'input_file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'input').iterdir())},
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'attempted_X_choices':len(Xs),'attempted_method_fits':len(Xs)*len(SPEC['methods']),
        'returned_scalar_estimates':len(estimates),'relationship_rows':len(relationships),
        'defined_intervals':int(estimates.defined.sum()),'failures':failures,
        'archive_reproduction_max_abs_differences':archive_checks,
        'previous_common_X_reproduction_max_abs_differences':previous_checks,
        'W_identical_across_X':True,'no_new_W_features':True}
    (ROOT/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'completed','X_choices':list(Xs),'all_rows':len(estimates),
                     'relationship_rows':len(relationships),'defined':int(estimates.defined.sum()),
                     'failures':len(failures),'max_archive_difference':max(archive_checks.values(), default=0.)}))


if __name__=='__main__':main()
