"""Ding et al. unbiased projection coefficients and conservative inference.

Unbiasedness concerns the coefficient vector, not its nonlinear plug-ins.
The Sxx normalization is n^{-1} X'X; finite covariance uses divisor n-1.
"""
import numpy as np
from finite import covariance, moments, unbiased_coefficients, _result, NAMES


def xy_vectors(X, Y):
    """Outcome-major order agrees with finite.moments and I2 kron Sxx^-1."""
    return np.concatenate([X * Y[:, j, None] for j in range(2)], axis=1)


def fit_ding(X, Y, Z):
    n, K = X.shape
    A = np.kron(np.eye(2), np.linalg.inv(X.T @ X / n))
    C = np.zeros((2 * K, 2 * K))
    for z in (0, 1):
        take = Z == z
        if take.sum() < 2:
            raise ValueError('Each arm requires two units for covariance.')
        q = xy_vectors(X[take], Y[take])
        C += A @ covariance(q) @ A.T / take.sum()
    return _result(X, unbiased_coefficients(X, Y, Z), C)


def ding_truth(X, Y0, Y1, p):
    n = len(X)
    Sinv = np.linalg.inv(X.T @ X / n)
    A = np.kron(np.eye(2), Sinv)
    beta = Sinv @ (X.T @ (Y1 - Y0) / n)
    estimate, G = moments(X, beta)
    q0, q1 = xy_vectors(X, Y0), xy_vectors(X, Y1)
    gap = A @ covariance(q1 - q0) @ A.T
    conservative = A @ (covariance(q1) / p + covariance(q0) / (1-p)) @ A.T
    sigma = conservative - gap
    return dict(names=NAMES.copy(), estimate=estimate, beta=beta, gradient=G,
                sigma2=np.einsum('ij,jk,ik->i', G, sigma, G),
                gap=np.einsum('ij,jk,ik->i', G, gap, G),
                beta_sigma=sigma, beta_gap=gap)
