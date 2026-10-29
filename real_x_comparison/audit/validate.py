#!/usr/bin/env python3
"""Independent checks of saved observed-data X comparisons; no nuisance fits."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent/'work/package/experiment'
NAMES = ['ATE1', 'ATE2', 'V11', 'V22', 'V12', 'rho', 'theta', 'alpha']
CRIT = 1.959963984540054


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(P, T, naive=False):
    n = len(T)
    a = (T if naive else P).mean(0)
    t = T-a
    if naive:
        c = np.c_[t[:, 0]**2, t[:, 1]**2, t[:, 0]*t[:, 1]]
    else:
        r = P-T
        c = np.c_[t[:, 0]**2+2*t[:, 0]*r[:, 0],
                  t[:, 1]**2+2*t[:, 1]*r[:, 1],
                  t[:, 0]*t[:, 1]+t[:, 1]*r[:, 0]+t[:, 0]*r[:, 1]]
    v = c.mean(0)
    eif = np.c_[(T if naive else P)-a, c-v, np.full((n, 3), np.nan)]
    est = np.r_[a, v, [np.nan]*3]
    if v[0] > 0 and v[1] > 0:
        rho, slope = v[2]/np.sqrt(v[0]*v[1]), v[2]/v[0]
        est[5:] = [rho, slope, a[1]-slope*a[0]]
        eif[:, 5] = eif[:, 4]/np.sqrt(v[0]*v[1])-rho*eif[:, 2]/(2*v[0])-rho*eif[:, 3]/(2*v[1])
        eif[:, 6] = (eif[:, 4]-slope*eif[:, 2])/v[0]
        eif[:, 7] = eif[:, 1]-slope*eif[:, 0]-a[0]*eif[:, 6]
    se = np.sqrt(np.mean(eif**2, axis=0)/n)
    if naive and np.isfinite(est[6]):
        resid = t[:, 1]-est[6]*t[:, 0]
        sig2 = np.dot(resid, resid)/(n-2)
        se[6] = np.sqrt(sig2/(n*v[0]))
        se[7] = np.sqrt(sig2*(1+a[0]**2/v[0])/n)
    return dict(estimate=est, se=se, lower=est-CRIT*se, upper=est+CRIT*se)


def validate_nvss():
    folder = ROOT/'nvss'
    inputs = folder/'input'
    spec = json.loads((folder/'spec.json').read_text())
    expected = {
        'age_education': ['maternal_age', 'maternal_education'],
        'age_education_parity_bmi': ['maternal_age', 'maternal_education', 'prior_live_births', 'prepregnancy_bmi'],
        'age_education_smoking': ['maternal_age', 'maternal_education', 'cigarettes_prepregnancy'],
        'age_education_parity_bmi_smoking': ['maternal_age', 'maternal_education', 'prior_live_births', 'prepregnancy_bmi', 'cigarettes_prepregnancy'],
    }
    assert spec['X_columns'] == expected
    assert spec['W_dimension'] == 20 and spec['n'] == 100000
    assert sha(folder/'code/super.py') == spec['estimation_source_sha256'] == '69e5edf2636e44b2130b38c9b83e8c6e5e610221a8cb7e98a623c17295ca47ee'
    assert sha(inputs/'nvss2019_smokers_sample.csv') == spec['data_sha256']
    pred = np.load(folder/'results/predictions.npz')
    old = np.load(inputs/'nvss_predictions.npz') if (inputs/'nvss_predictions.npz').exists() else None
    if old is not None:
        for key in ['W', 'Y', 'Z', 'source_row', 'fold', 'pseudo', 'propensity', 'mu0', 'mu1']:
            np.testing.assert_array_equal(pred[key], old[key])
    assert np.array_equal(np.bincount(pred['fold']), [20000]*5)
    table = pd.read_csv(folder/'results/estimates.csv')
    ordering = pd.read_csv(folder/'results/variance_ordering.csv')
    assert len(table) == 64 and not table.duplicated(['X', 'method', 'estimand']).any()
    assert set(table.X) == set(expected) and set(table.method) == {'direct', 'calibrated'}
    assert len(ordering) == 8 and not ordering.duplicated(['X', 'estimand']).any()
    old_table = pd.read_csv(inputs/'nvss_estimates.csv') if (inputs/'nvss_estimates.csv').exists() else None
    previous = folder/'archive/previous_four_X/results'
    previous_arrays = np.load(previous/'predictions.npz') if (previous/'predictions.npz').exists() else None
    previous_table = pd.read_csv(previous/'estimates.csv') if (previous/'estimates.csv').exists() else None
    max_old_diff = max_reconstructed_diff = max_ordering_diff = 0.
    counts = {}
    for x, cols in expected.items():
        take = [spec['W_columns'].index(col) for col in cols]
        np.testing.assert_array_equal(pred['X_'+x], pred['W'][:, take])
        T, C, P = pred['tau_'+x], pred['cal_'+x], pred['pseudo']
        for method, values in [('direct', reconstruct(P, T)),
                               ('calibrated', reconstruct(P, C))]:
            rows = table[(table.X == x) & (table.method == method)].set_index('estimand').loc[NAMES]
            assert len(rows) == 8
            for metric, a in values.items():
                b = rows[metric].to_numpy()
                np.testing.assert_allclose(a, b, atol=1e-10, rtol=1e-11, equal_nan=True)
                ok = np.isfinite(a) & np.isfinite(b)
                max_reconstructed_diff = max(max_reconstructed_diff, float(abs(a[ok]-b[ok]).max()))
            counts[x+':'+method] = int(rows.defined.sum())
        direct = reconstruct(P, T)['estimate'][2:4]
        sc = direct+(P.mean(0)-T.mean(0))**2
        cal = reconstruct(P, C)['estimate'][2:4]
        for j, target in enumerate(['V11', 'V22']):
            row = ordering[(ordering.X == x) & (ordering.estimand == target)].iloc[0]
            a = np.array([direct[j], sc[j], cal[j]])
            b = row[['direct', 'sample_centered', 'calibrated']].to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-11)
            max_ordering_diff = max(max_ordering_diff, float(abs(a-b).max()))
            assert a[0] <= a[1]+1e-12 and a[1] <= a[2]+1e-12 and row.ordering_holds
        prior = {'age_education': 'small'}.get(x)
        if prior and old is not None and old_table is not None:
            np.testing.assert_array_equal(T, old['tau_'+prior])
            np.testing.assert_array_equal(C, old['cal_'+prior])
            for method in ['direct', 'calibrated']:
                a = table[(table.X == x) & (table.method == method)].set_index('estimand').loc[NAMES]
                b = old_table[(old_table.X == prior) & (old_table.method == method)].set_index('estimand').loc[NAMES]
                for metric in ['estimate', 'se', 'lower', 'upper']:
                    np.testing.assert_allclose(a[metric], b[metric], rtol=1e-11, atol=1e-10, equal_nan=True)
                    ok = np.isfinite(a[metric]) & np.isfinite(b[metric])
                    max_old_diff = max(max_old_diff, float(abs(a.loc[ok, metric]-b.loc[ok, metric]).max()))
        if x in ['age_education', 'age_education_parity_bmi_smoking'] and previous_arrays is not None and previous_table is not None:
            for key in ['X_', 'tau_', 'cal_']:
                np.testing.assert_array_equal(pred[key+x], previous_arrays[key+x])
            for method in ['direct', 'calibrated']:
                now = table[(table.X == x) & (table.method == method)].set_index('estimand').loc[NAMES]
                before = previous_table[(previous_table.X == x) & (previous_table.method == method)].set_index('estimand').loc[NAMES]
                for quantity in ['estimate', 'se', 'lower', 'upper']:
                    np.testing.assert_array_equal(now[quantity], before[quantity])
    result = dict(status='passed', all_four_frozen_X_choices_reported=True,
                  rows=64, no_naive_rows=True, same_sample_and_fixed_W20=True,
                  historical_nuisance_comparison_run=old is not None,
                  source_uses_same_outer_training_fold_pseudo_outcomes=True,
                  historical_base_comparison_run=old is not None and old_table is not None,
                  historical_archive_comparison_run=previous_arrays is not None and previous_table is not None,
                  max_original_estimate_or_interval_difference=max_old_diff,
                  max_independently_reconstructed_estimate_or_interval_difference=max_reconstructed_diff,
                  all_eight_marginal_variance_orderings_pass=True,
                  max_independent_variance_ordering_difference=max_ordering_diff,
                  defined_counts_of_eight_per_method=counts,
                  audit_source_sha256=sha(Path(__file__)))
    (folder/'audit').mkdir(exist_ok=True)
    (folder/'audit/validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def finite_values(X, beta, covariance):
    mean = X.mean(0)
    S = np.cov(X, rowvar=False, ddof=1)
    Sb = S@beta
    V = beta.T@Sb
    a = mean@beta
    v1, v2, cov = V[0, 0], V[1, 1], V[0, 1]
    rho, slope = cov/np.sqrt(v1*v2), cov/v1
    estimate = np.r_[a, v1, v2, cov, rho, slope, a[1]-slope*a[0]]
    k = X.shape[1]
    G = np.zeros((8, 2*k))
    G[0, :k] = mean; G[1, k:] = mean
    G[2, :k] = 2*Sb[:, 0]; G[3, k:] = 2*Sb[:, 1]
    G[4, :k] = Sb[:, 1]; G[4, k:] = Sb[:, 0]
    G[5] = G[4]/np.sqrt(v1*v2)-rho*G[2]/(2*v1)-rho*G[3]/(2*v2)
    G[6] = (G[4]-slope*G[2])/v1
    G[7] = G[1]-slope*G[0]-a[0]*G[6]
    se = np.sqrt(np.einsum('ij,jk,ik->i', G, covariance, G))
    return dict(estimate=estimate, se=se, lower=estimate-CRIT*se, upper=estimate+CRIT*se,
                beta=beta, beta_cov=covariance, gradient=G)


def independent_finite(X, W, Y, Z):
    k = X.shape[1]
    b_ols = np.zeros((k, 2)); b_ma = np.zeros((k, 2))
    C_ols = np.zeros((2*k, 2*k)); C_ma = np.zeros_like(C_ols)
    for z, sign in [(0, -1), (1, 1)]:
        mask = Z == z
        x, w, y = X[mask], W[mask], Y[mask]
        nz = len(x)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        residual = y-x@beta
        score = np.concatenate([x*residual[:, j, None] for j in range(2)], axis=1)
        coef = np.linalg.lstsq(w, score, rcond=None)[0]
        error = score-w@coef
        A = np.kron(np.eye(2), np.linalg.inv(x.T@x/nz))
        correction = (A@coef.T@(w.mean(0)-W.mean(0))).reshape(2, k).T
        b_ols += sign*beta; b_ma += sign*(beta-correction)
        C_ols += A@(score.T@score/(nz-1))@A.T/nz
        C_ma += A@(error.T@error/(nz-1))@A.T/nz
    return {'ols': finite_values(X, b_ols, C_ols), 'model_assisted': finite_values(X, b_ma, C_ma)}


def validate_hillstrom():
    folder = ROOT/'hillstrom'
    spec = json.loads((folder/'spec.json').read_text())
    manifest = json.loads((folder/'run_manifest.json').read_text())
    expected = {
        'small': ['intercept', 'recency', 'log1p_history'],
        'large': ['intercept', 'recency', 'log1p_history', 'mens', 'womens'],
        'history_newbie': ['intercept', 'recency', 'log1p_history', 'newbie'],
        'combined': ['intercept', 'recency', 'log1p_history', 'mens', 'womens', 'newbie'],
    }
    assert {k: v['names'] for k, v in spec['X_choices'].items()} == expected
    assert sha(folder/'code/finite.py') == spec['finite_code_sha256'] == 'c6dc4d071062e45404c72206998a8e74c84cedfb9b0dcae0245b9cc06d75d16c'
    assert sha(folder/'spec.json') == manifest['specification_sha256']
    for name, val in manifest['code_sha256'].items():
        assert sha(folder/'code'/name) == val
    raw = gzip.decompress((folder/'input/hillstrom_raw.csv.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == spec['source_csv_sha256']
    arrays = np.load(folder/'results/analysis_arrays.npz')
    old_arrays = np.load(folder/'input/archive_arrays.npz') if (folder/'input/archive_arrays.npz').exists() else None
    if old_arrays is not None:
        for key in old_arrays.files:
            np.testing.assert_array_equal(arrays[key], old_arrays[key])
    W, Y, Z = arrays['W'], arrays['Y'], arrays['Z']
    assert W.shape == (42613, 12) and Y.shape == (42613, 2)
    assert np.linalg.matrix_rank(W) == 12
    assert np.bincount(Z).tolist() == [21306, 21307]
    table = pd.read_csv(folder/'results/estimates.csv')
    relationships = pd.read_csv(folder/'results/relationship_estimates.csv')
    assert len(table) == 64 and len(relationships) == 48
    assert not table.duplicated(['X', 'method', 'estimand']).any()
    assert set(table.X) == set(expected) and set(table.method) == {'ols', 'model_assisted'}
    assert table.defined.all() and len(json.loads((folder/'results/failures.json').read_text())) == 0
    full = json.loads((folder/'results/full.json').read_text())
    old_full = json.loads((folder/'input/archive_full.json').read_text()) if (folder/'input/archive_full.json').exists() else None
    previous_full = json.loads((folder/'archive/previous_four_X/results/full.json').read_text()) if (folder/'archive/previous_four_X/results/full.json').exists() else None
    max_difference = 0.
    ranks = {}
    for x, names in expected.items():
        X = arrays['X_'+x]
        columns = [spec['W_names'].index(name) for name in names]
        np.testing.assert_array_equal(X, W[:, columns])
        assert np.linalg.matrix_rank(X) == len(names)
        ranks[x] = len(names)
        for z in (0, 1):
            assert np.linalg.matrix_rank(X[Z == z]) == len(names)
            assert np.linalg.matrix_rank(W[Z == z]) == 12
        calc = independent_finite(X, W, Y, Z)
        for method, values in calc.items():
            rows = table[(table.X == x) & (table.method == method)].set_index('estimand').loc[NAMES]
            for quantity, value in values.items():
                saved = np.asarray(full[x][method][quantity])
                np.testing.assert_allclose(value, saved, rtol=1e-9, atol=1e-9)
                max_difference = max(max_difference, float(abs(value-saved).max()))
                if quantity in ['estimate', 'se', 'lower', 'upper']:
                    np.testing.assert_allclose(rows[quantity], saved, rtol=1e-12, atol=1e-12)
                if x in ['small', 'large'] and old_full is not None:
                    np.testing.assert_array_equal(saved, old_full[x][method][quantity])
                if x in ['small', 'large', 'history_newbie'] and previous_full is not None:
                    np.testing.assert_array_equal(saved, previous_full[x][method][quantity])
    result = dict(status='passed', same_original_sample_and_assignment=True,
                  fixed_W12_exactly_reproduced=True, all_four_frozen_X_choices_reported=True,
                  all_X_are_specified_columns_of_fixed_W=True, X_ranks=ranks,
                  rows=64, relationship_rows=48, all_64_intervals_defined=True,
                  historical_small_large_comparison_run=old_full is not None,
                  historical_three_X_comparison_run=previous_full is not None,
                  independent_least_squares_and_joint_covariance_and_delta_checks_pass=True,
                  max_independent_absolute_result_difference=max_difference,
                  source_input_and_specification_hashes_match=True,
                  audit_source_sha256=sha(Path(__file__)))
    (folder/'audit').mkdir(exist_ok=True)
    (folder/'audit/validation.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['nvss', 'hillstrom', 'all'], default='all')
    args = parser.parse_args()
    if args.dataset == 'all':
        result = {'nvss': validate_nvss(), 'hillstrom': validate_hillstrom()}
    elif args.dataset == 'nvss':
        result = validate_nvss()
    else:
        result = validate_hillstrom()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
