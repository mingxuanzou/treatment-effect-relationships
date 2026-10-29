"""Exhaustive balanced randomization check of Ding covariance normalizations."""
from itertools import combinations
import numpy as np
from ding import fit_ding, ding_truth
from finite import fit_finite


def exhaustive_check():
    n = 8
    x = np.array([-1.8, -1.3, -.6, -.2, .3, .7, 1.2, 1.7])
    X = np.c_[np.ones(n), x]
    Y0 = np.c_[.5 + .4*x + .3*x*x, -.8 + .7*x + np.sin(x)]
    Y1 = Y0 + np.c_[1 + .5*x + .2*np.cos(x), -.4 - .3*x + .15*x*x]
    truth = ding_truth(X, Y0, Y1, .5)
    betas, Cs = [], []
    shift_max_ols, shift_max_ding = 0., 0.
    for treated in combinations(range(n), n//2):
        Z = np.zeros(n, dtype=int)
        Z[list(treated)] = 1
        Y = np.where(Z[:, None], Y1, Y0)
        f = fit_ding(X, Y, Z)
        betas.append(f['beta'].T.ravel())
        Cs.append(f['beta_cov'])
        # A common outcome shift leaves targets fixed, but affects raw XY imbalance.
        ols = fit_finite(X, Y, Z)['ols']['beta']
        ols_shift = fit_finite(X, Y + 2., Z)['ols']['beta']
        ding_shift = fit_ding(X, Y + 2., Z)['beta']
        shift_max_ols = max(shift_max_ols, float(np.max(abs(ols_shift - ols))))
        shift_max_ding = max(shift_max_ding, float(np.max(abs(ding_shift - f['beta']))))
    betas, Cs = np.array(betas), np.array(Cs)
    mean_error = float(np.max(abs(betas.mean(0) - truth['beta'].T.ravel())))
    exact_cov = np.cov(betas, rowvar=False, ddof=0)
    cov_error = float(np.max(abs(exact_cov - truth['beta_sigma']/n)))
    conservative_error = float(np.max(abs(Cs.mean(0) - (truth['beta_sigma']+truth['beta_gap'])/n)))
    np.testing.assert_allclose(betas.mean(0), truth['beta'].T.ravel(), atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(exact_cov, truth['beta_sigma']/n, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(Cs.mean(0), (truth['beta_sigma']+truth['beta_gap'])/n, atol=1e-13, rtol=1e-13)
    assert shift_max_ols < 1e-12
    assert shift_max_ding > 1e-3
    return dict(assignments=len(betas), n=n, coefficient_mean_max_error=mean_error,
                exact_covariance_max_error=cov_error, mean_conservative_covariance_max_error=conservative_error,
                common_shift_ols_max_coefficient_change=shift_max_ols,
                common_shift_ding_max_coefficient_change=shift_max_ding)


if __name__ == '__main__':
    import json
    print(json.dumps(exhaustive_check(), indent=2))
