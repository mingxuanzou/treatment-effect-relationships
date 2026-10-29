#!/usr/bin/env python3
"""Compare four fixed reduced-CATE definitions in the observed NVSS sample."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key] = '1'
from pathlib import Path
from io import BytesIO
import hashlib
import json
import time
import zipfile
import numpy as np
import pandas as pd
from super import fit_super, NAMES

OUT = Path(__file__).resolve().parents[1]
INPUT = OUT/'input'
ARCHIVE = OUT/'archive/previous_four_X'
X_COLUMNS = {
    'age_education':['maternal_age','maternal_education'],
    'age_education_parity_bmi':['maternal_age','maternal_education','prior_live_births','prepregnancy_bmi'],
    'age_education_smoking':['maternal_age','maternal_education','cigarettes_prepregnancy'],
    'age_education_parity_bmi_smoking':['maternal_age','maternal_education','prior_live_births','prepregnancy_bmi','cigarettes_prepregnancy'],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    start=time.time()
    (OUT/'results').mkdir(exist_ok=True)
    data_path=INPUT/'nvss2019_smokers_sample.csv'
    d=pd.read_csv(data_path)
    nums=['maternal_age','maternal_education','prior_live_births','prepregnancy_bmi','cigarettes_prepregnancy','prepregnancy_diabetes','prepregnancy_hypertension','male_infant']
    cats=['maternal_nativity','maternal_race6','maternal_hispanic_origin']
    wframe=pd.concat([d[nums],pd.get_dummies(d[cats].astype(str),drop_first=True,dtype=float)],axis=1)
    W=wframe.to_numpy(float)
    X={name:d[cols].to_numpy(float) for name,cols in X_COLUMNS.items()}
    Y=d[['birthweight_grams','gestation_weeks']].to_numpy(float)/[100.,1.]
    Z=d.Z.to_numpy(int)
    spec={
        'dataset':'NVSS2019','analysis':'Observed-data exploratory comparison of four fixed common-base X choices',
        'X_choices_frozen_before_this_comparison':True,'X_columns':X_COLUMNS,
        'X_design':{'base':['maternal_age','maternal_education'],'block_A':['prior_live_births','prepregnancy_bmi'],'block_B':['cigarettes_prepregnancy'],'order':['base','base + block A','base + block B','base + blocks A and B']},
        'W_columns':list(wframe.columns),'W_dimension':W.shape[1],'W_fixed_across_X':True,
        'Y':['birth weight in 100 grams','gestational duration in weeks'],
        'Z':'First-trimester smoking continuation (1) versus cessation (0) among prepregnancy smokers',
        'sample':'Same existing primary sample of 100000 complete singleton live-birth records to mothers age 20–44',
        'n':len(Z),'n1':int(Z.sum()),'fold_seed':2019102901,'folds':5,
        'learner':'Componentwise least-squares gradient boosting of regression stumps',
        'n_estimators':500,'learning_rate':.1,'propensity_truncation':[.01,.99],
        'cate_prediction_clip':[30.,20.],'methods':['direct','calibrated'],
        'inference':'Original EIF-based untruncated pointwise 95% Wald intervals',
        'shared_nuisance_fits':'Full W nuisances are refit once per original outer fold and shared across every X. Reduced-CATE fits use training-fold pseudo-outcomes from those same training-fold nuisance fits, not cached held-out pseudo-outcomes.',
        'h':'Maternal age >=30; passed explicitly to retain original internal diagnostic; main reported calibrated method excludes h.',
        'data_sha256':sha(data_path),'estimation_source_sha256':sha(Path(__file__).with_name('super.py')),
        'interpretation':'Changing X changes the reduced-CATE target; this is an exploratory target comparison, not repeated-sampling coverage or a search for a preferred sign.',
    }
    (OUT/'spec.json').write_text(json.dumps(spec,indent=2))
    print('Specification frozen for four X choices.',flush=True)
    old=np.load(INPUT/'nvss_predictions.npz') if (INPUT/'nvss_predictions.npz').exists() else None
    if old is not None:
        for name,array in [('W',W),('Y',Y),('Z',Z)]:
            np.testing.assert_array_equal(array,old[name])
    fit_start=time.time()
    result=fit_super(W,X,Y,Z,seed=2019102901,n_estimators=500,clip=[30.,20.],return_folds=True,h=d.maternal_age.to_numpy()>=30)
    fit_elapsed=time.time()-fit_start
    print('All shared fits completed in',round(fit_elapsed,3),'seconds.',flush=True)
    rows=[];order=[]
    predictions={'W':W,'Y':Y,'Z':Z,'source_row':d.source_row.to_numpy(),
                 'pseudo':result['age_education']['pseudo'],'fold':result['age_education']['fold'],
                 **result['age_education']['nuisance']}
    for x,r in result.items():
        predictions['X_'+x]=X[x]
        predictions['tau_'+x]=r['tau']
        predictions['cal_'+x]=r['calibrated_tau']
        for method in spec['methods']:
            for j,target in enumerate(NAMES):
                q=r[method]
                rows.append(dict(dataset='NVSS2019',X=x,method=method,estimand=target,
                                 **{key:float(q[key][j]) for key in ['estimate','se','lower','upper']},
                                 defined=bool(np.isfinite([q['estimate'][j],q['se'][j]]).all()),n=len(Z)))
        for j,target in enumerate(['V11','V22']):
            vals=[float(r['direct']['estimate'][j+2]),float(r['sample_centered_cov'][j]),float(r['calibrated']['estimate'][j+2])]
            order.append(dict(X=x,estimand=target,direct=vals[0],sample_centered=vals[1],calibrated=vals[2],ordering_holds=bool(vals[0]<=vals[1]+1e-12 and vals[1]<=vals[2]+1e-12)))
    estimates=pd.DataFrame(rows)
    estimates.to_csv(OUT/'results/estimates.csv',index=False)
    pd.DataFrame(order).to_csv(OUT/'results/variance_ordering.csv',index=False)
    buffer=BytesIO()
    np.savez_compressed(buffer,**predictions)
    payload=buffer.getvalue()
    with zipfile.ZipFile(BytesIO(payload)) as saved_zip:
        assert saved_zip.testzip() is None
    target=OUT/'results/predictions.npz'
    temporary=target.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    with zipfile.ZipFile(target) as saved_zip:
        assert saved_zip.testzip() is None
    with np.load(target) as saved_predictions:
        assert set(saved_predictions.files)==set(predictions)
        for key,expected in predictions.items():
            np.testing.assert_array_equal(saved_predictions[key],expected)
    checks={}
    maxdiff=0.
    if old is not None and (INPUT/'nvss_estimates.csv').exists():
        for key in ['W','Y','Z','source_row','fold','pseudo','propensity','mu0','mu1']:
            np.testing.assert_array_equal(predictions[key],old[key])
            checks[f'original:{key}:exactly_equal']=True
        old_table=pd.read_csv(INPUT/'nvss_estimates.csv')
        for now,previous in [('age_education','small')]:
            r=result[now]
            for key,oldkey in [('tau','tau_'+previous),('calibrated_tau','cal_'+previous),('pseudo','pseudo'),('fold','fold')]:
                diff=float(np.max(abs(r[key]-old[oldkey])))
                checks[f'{now}:{key}:max_abs_difference']=diff
                np.testing.assert_allclose(r[key],old[oldkey],atol=1e-10,rtol=1e-10)
                maxdiff=max(maxdiff,diff)
            for method in spec['methods']:
                q=old_table[(old_table.X==previous)&(old_table.method==method)].set_index('estimand').loc[NAMES]
                for metric in ['estimate','se','lower','upper']:
                    a=r[method][metric];b=q[metric].to_numpy()
                    np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True)
                    finite=np.isfinite(a)&np.isfinite(b)
                    diff=float(np.max(abs(a[finite]-b[finite])))
                    checks[f'{now}:{method}:{metric}:max_abs_difference']=diff
                    maxdiff=max(maxdiff,diff)
    if (ARCHIVE/'results/predictions.npz').exists() and (ARCHIVE/'results/estimates.csv').exists():
        previous_predictions=np.load(ARCHIVE/'results/predictions.npz')
        previous_table=pd.read_csv(ARCHIVE/'results/estimates.csv')
        for x in ['age_education','age_education_parity_bmi_smoking']:
            for prefix in ['X_','tau_','cal_']:
                key=prefix+x
                np.testing.assert_array_equal(predictions[key],previous_predictions[key])
                checks[f'previous_comparison:{key}:exactly_equal']=True
            for method in spec['methods']:
                q=previous_table[(previous_table.X==x)&(previous_table.method==method)].set_index('estimand').loc[NAMES]
                for metric in ['estimate','se','lower','upper']:
                    a=result[x][method][metric];b=q[metric].to_numpy()
                    np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12,equal_nan=True)
                    finite=np.isfinite(a)&np.isfinite(b)
                    diff=float(np.max(abs(a[finite]-b[finite])))
                    checks[f'previous_comparison:{x}:{method}:{metric}:max_abs_difference']=diff
                    maxdiff=max(maxdiff,diff)
    assert len(estimates)==64
    assert all(row['ordering_holds'] for row in order)
    valid={
        'status':'passed','historical_input_checks_run':old is not None,
        'historical_archive_checks_run':(ARCHIVE/'results/predictions.npz').exists(),'max_difference_from_available_references':maxdiff,
        'checks':checks,'marginal_variance_ordering_holds_for_all_X':True,
        'all_four_fixed_choices_reported':True,'rows':len(estimates),
        'fit_elapsed_seconds':fit_elapsed,'elapsed_seconds':time.time()-start,
    }
    (OUT/'validation.json').write_text(json.dumps(valid,indent=2))
    print(json.dumps({key:valid[key] for key in ['status','max_difference_from_available_references','elapsed_seconds']}),flush=True)
    print(estimates[(estimates.method=='calibrated')&estimates.estimand.isin(['V11','V22','V12','rho','theta','alpha'])][['X','estimand','estimate','lower','upper']].to_string(index=False),flush=True)

if __name__=='__main__':
    run()
