"""Revision-4 bounded benchmark: modestly regularized cubic-augmented CATE learner.

The inference core is unchanged from the archived first experiment. This module
changes only the data-generating process and nuisance learning procedures.
"""
import os
for _key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'
import sys
from pathlib import Path
import itertools
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from super import from_predictions, NAMES, ZCRIT

LEARNERS = ('series', 'poly_boost', 'spline', 'tree')
SUPPORTED_LEARNERS = LEARNERS + ('rate_ridge', 'cubic_ridge')
SQRT3 = np.sqrt(3.)


def q(x):
    return (np.asarray(x)**2 - 1) / np.sqrt(.8)


def indices(W):
    x1, x2 = W[:, 0], W[:, 1]
    return np.column_stack(((x1 + .6*q(x2))/np.sqrt(1.36),
                            (-.6*x1 + q(x2))/np.sqrt(1.36), W[:, 2]))


def coefficients(rho=.5, s1=.8, s2=1.1):
    a = (rho-.36)/.8
    if .64-a*a < -1e-12:
        raise ValueError('This shared-component family requires rho in [-.28, 1].')
    b = np.sqrt(max(0., .64-a*a))
    return np.array([[.8*s1, 0., .6*s1], [a*s2, b*s2, .6*s2]])


def truth(rho=.5, s1=.8, s2=1.1, nested='large'):
    k = 2 if nested == 'small' else 3
    B = coefficients(rho, s1, s2)[:, :k]
    V = B @ B.T
    theta = V[0, 1]/V[0, 0]
    return np.array([1., -.5, V[0, 0], V[1, 1], V[0, 1],
                     V[0, 1]/np.sqrt(V[0, 0]*V[1, 1]), theta, -.5-theta])


def components(W, rho=.5, s1=.8, s2=1.1, design='observational',
               noise_corr=.3, common=0.):
    x1, x2, x3, u = np.asarray(W).T
    basis = indices(W)
    B = coefficients(rho, s1, s2)
    tau = np.array([1., -.5]) + basis @ B.T
    full = tau + u[:, None]*np.array([.3, -.25])
    m0 = np.column_stack((.6*x1 + .3*q(x2) + .4*q(x3) + .8*u + .2*x1*u,
                          .4*x2 + .4*q(x1) + .2*q(x3) + .7*u - .15*x2*u))
    m0 += common*(x1+u)[:, None]
    e = np.full(len(W), .5) if design == 'randomized' else .5+.06*x1-.04*q(x2)+.06*u
    reduced = {'large': tau,
               'small': np.array([1., -.5]) + basis[:, :2] @ B[:, :2].T}
    return {'mu0': m0, 'mu1': m0+full, 'fulltau': full, 'e': e,
            'tau': reduced, 'B': B}


def dgp(n, seed, **settings):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-SQRT3, SQRT3, (n, 4))
    c = components(W, **settings)
    Z = rng.binomial(1, c['e'])
    nc = settings.get('noise_corr', .3)
    errors = rng.multivariate_normal([0., 0.], [[1., nc], [nc, 1.]], n)
    Y = c['mu0'] + Z[:, None]*c['fulltau'] + errors
    return W, Y, Z, c


def polynomial_design(X):
    """Full degree-two basis without intercept; centered squares improve scaling."""
    X = np.asarray(X, float)
    cols = [X, q(X)]
    if X.shape[1] > 1:
        cols.append(np.column_stack([X[:, j]*X[:, k] for j in range(X.shape[1])
                                     for k in range(j+1, X.shape[1])]))
    return np.column_stack(cols)


class LinearFeatureRegressor:
    def __init__(self, learner='series'):
        self.learner = learner

    def features(self, X, fit=False):
        X = np.asarray(X, float)
        if self.learner == 'cubic_ridge':
            z = X / SQRT3
            cubic = np.sqrt(7.) * (5*z**3 - 3*z) / 2
            return np.column_stack((polynomial_design(X), cubic))
        if self.learner != 'spline':
            return polynomial_design(X)
        if fit:
            # Fixed bounded-domain knots avoid any outcome-dependent knot choice.
            knots = np.tile(np.linspace(-SQRT3, SQRT3, 5)[:, None], (1, X.shape[1]))
            self.spline = SplineTransformer(knots=knots, degree=3, include_bias=False,
                                            extrapolation='linear')
            A = self.spline.fit_transform(X)
        else:
            A = self.spline.transform(X)
        # The true interaction terms are included; univariate components are flexible.
        pair = [X[:, j]*X[:, k] for j in range(X.shape[1])
                for k in range(j+1, X.shape[1])]
        return np.column_stack([A] + pair)

    def fit(self, X, Y):
        Y = np.asarray(Y, float)
        self.scalar = Y.ndim == 1
        if self.scalar:
            Y = Y[:, None]
        F = self.features(X, fit=True)
        self.mean = F.mean(0)
        self.scale = np.maximum(F.std(0), 1e-8)
        A = (F-self.mean)/self.scale
        self.intercept = Y.mean(0)
        G = A.T@A
        R = A.T@(Y-self.intercept)
        if self.learner == 'poly_boost':
            # Componentwise L2 gradient boosting with linear polynomial weak learners.
            # Iterations grow with n; neither loss nor coverage chooses the stopping point.
            self.rounds = max(500, int(np.ceil(200*np.log(max(len(X), 2)))))
            self.coef = np.zeros_like(R)
            D = np.maximum(np.diag(G), 1e-12)
            for _ in range(self.rounds):
                j = np.argmax(R*R/D[:, None], axis=0)
                step = .1*R[j, np.arange(Y.shape[1])]/D[j]
                self.coef[j, np.arange(Y.shape[1])] += step
                R -= G[:, j]*step
        else:
            if self.learner in ('rate_ridge', 'cubic_ridge'):
                # A separate, controlled reduced-CATE rate illustration.
                # Objective: mean squared error + lambda_n sum_j weight_j beta_j^2.
                # Linear terms receive weight 1; squares and interactions weight 2.
                penalty = np.full(G.shape[0], 2.)
                penalty[:np.asarray(X).shape[1]] = 1.
                self.lambda_n = (0.3 if self.learner == 'cubic_ridge' else 2.)*len(X)**(-.3)
                self.coef = np.linalg.solve(G+len(X)*self.lambda_n*np.diag(penalty), R)
            else:
                alpha = 1e-6 if self.learner == 'series' else 1e-4
                self.coef = np.linalg.solve(G+alpha*np.eye(G.shape[0]), R)
        return self

    def predict(self, X):
        y = self.intercept + ((self.features(X)-self.mean)/self.scale)@self.coef
        return y[:, 0] if self.scalar else y


class TreeRegressor:
    def fit(self, X, Y):
        Y = np.asarray(Y, float)
        self.scalar = Y.ndim == 1
        if self.scalar:
            Y = Y[:, None]
        self.models = []
        for j in range(Y.shape[1]):
            model = HistGradientBoostingRegressor(loss='squared_error', learning_rate=.08,
                     max_iter=150, max_leaf_nodes=15, min_samples_leaf=20,
                     l2_regularization=1., early_stopping=False, random_state=0)
            self.models.append(model.fit(X, Y[:, j]))
        return self

    def predict(self, X):
        y = np.column_stack([m.predict(X) for m in self.models])
        return y[:, 0] if self.scalar else y


def make_learner(learner):
    if learner not in SUPPORTED_LEARNERS:
        raise ValueError(f'Unknown learner: {learner}')
    return TreeRegressor() if learner == 'tree' else LinearFeatureRegressor(learner)


def fit_super_v2(W, X, Y, Z, seed=20261028, known_e=None, learner='series',
                 clip=12., return_folds=True, h=None):
    W, Y, Z = np.asarray(W, float), np.asarray(Y, float), np.asarray(Z, int)
    n = len(W)
    if not np.isfinite(W).all() or not np.isfinite(Y).all():
        raise ValueError('Complete finite W and Y required.')
    xs = X if isinstance(X, dict) else {'X': np.asarray(X, float)}
    fold = np.empty(n, dtype=int)
    fold[np.random.default_rng(seed).permutation(n)] = np.arange(n) % 5
    P = np.empty_like(Y)
    TT = {key: np.empty_like(Y) for key in xs}
    ep, eraw = np.empty(n), np.empty(n)
    mu0, mu1 = np.empty_like(Y), np.empty_like(Y)
    eall = None if known_e is None else np.broadcast_to(known_e, (n,))
    for k in range(5):
        tr, te = fold != k, fold == k
        wt, yt, zt = W[tr], Y[tr], Z[tr]
        nuisance_learner = 'series' if learner in ('rate_ridge', 'cubic_ridge') else learner
        m0 = make_learner(nuisance_learner).fit(wt[zt == 0], yt[zt == 0])
        m1 = make_learner(nuisance_learner).fit(wt[zt == 1], yt[zt == 1])
        p0, p1 = m0.predict(W[te]), m1.predict(W[te])
        u0, u1 = m0.predict(wt), m1.predict(wt)
        if known_e is None:
            pe = make_learner(nuisance_learner).fit(wt, zt)
            raw, rt = pe.predict(W[te]), pe.predict(wt)
            ee, et = np.clip(raw, .01, .99), np.clip(rt, .01, .99)
        else:
            raw = ee = eall[te]
            et = eall[tr]
        P[te] = p1-p0+(Z[te]/ee)[:, None]*(Y[te]-p1)-((1-Z[te])/(1-ee))[:, None]*(Y[te]-p0)
        ptr = u1-u0+(zt/et)[:, None]*(yt-u1)-((1-zt)/(1-et))[:, None]*(yt-u0)
        ep[te], eraw[te], mu0[te], mu1[te] = ee, raw, p0, p1
        for key, xx in xs.items():
            xx = np.asarray(xx, float)
            tt = make_learner(learner).fit(xx[tr], ptr).predict(xx[te])
            TT[key][te] = np.clip(tt, -clip, clip)
    out = {}
    for key, xx in xs.items():
        hh = (np.asarray(xx)[:, 0]*np.asarray(xx)[:, 1] > 0) if h is None else h
        rr = from_predictions(P, TT[key], h=hh)
        rr['nuisance'] = {'propensity': ep, 'propensity_raw': eraw, 'mu0': mu0, 'mu1': mu1}
        rr['diagnostics'].update(propensity_min=float(ep.min()), propensity_max=float(ep.max()),
                    propensity_truncated_fraction=float(np.mean((eraw < .01) | (eraw > .99))),
                    cate_clipped_fraction=float(np.mean(np.abs(TT[key]) >= clip)))
        if return_folds:
            rr['fold'] = fold
        out[key] = rr
    return out if isinstance(X, dict) else out['X']


def efficiency(settings, nested, order=12):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    ids = np.array(list(itertools.product(range(order), repeat=4)))
    W = SQRT3*nodes[ids]
    ww = np.prod(weights[ids]/2, axis=1)
    c = components(W, **settings)
    target = truth(settings.get('rho', .5), settings.get('s1', .8), settings.get('s2', 1.1), nested)
    a, v = target[:2], target[2:5]
    tc = c['tau'][nested]-a
    delta = c['fulltau']-c['tau'][nested]
    f = np.column_stack((c['fulltau']-a,
        tc[:, 0]**2+2*tc[:, 0]*delta[:, 0]-v[0],
        tc[:, 1]**2+2*tc[:, 1]*delta[:, 1]-v[1],
        tc[:, 0]*tc[:, 1]+tc[:, 1]*delta[:, 0]+tc[:, 0]*delta[:, 1]-v[2]))
    L = np.zeros((len(W), 5, 2))
    L[:, 0, 0], L[:, 1, 1] = 1, 1
    L[:, 2, 0], L[:, 3, 1] = 2*tc[:, 0], 2*tc[:, 1]
    L[:, 4, 0], L[:, 4, 1] = tc[:, 1], tc[:, 0]
    noise_corr = settings.get('noise_corr', .3)
    S = np.array([[1., noise_corr], [noise_corr, 1.]])
    cov = f.T@(ww[:, None]*f)+np.einsum('n,nik,kl,njl->ij', ww/(c['e']*(1-c['e'])), L, S, L)
    J = np.zeros((8, 5)); J[:5, :5] = np.eye(5)
    v1, v2, v12 = v; rho, theta = target[5:7]
    J[5, 2:] = [-rho/(2*v1), -rho/(2*v2), 1/np.sqrt(v1*v2)]
    J[6, 2:] = [-theta/v1, 0, 1/v1]
    J[7] = np.array([-theta, 1, 0, 0, 0])-a[0]*J[6]
    return np.sqrt(np.einsum('ik,kl,il->i', J, cov, J))


fit_super = fit_super_v2
