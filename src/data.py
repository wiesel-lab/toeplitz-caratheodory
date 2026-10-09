"""Synthetic data: the random ensemble of true covariances (Figures 2 and 3)
and Gaussian snapshots with their sample covariance (Figure 3)."""

import numpy as np
from numpy.linalg import cholesky

from src.caratheodory import steering_matrix


def random_covariance(P, rng):
    """Full-rank Toeplitz covariance built from P atoms plus the noise term 0.15 I.

    The frequencies are uniform in [0, 2 pi) and the amplitudes uniform in
    [0.5, A], where A is drawn uniformly from [2, 5] for each covariance.
    """
    AA = float(rng.uniform(2.0, 5.0))
    amps = rng.uniform(0.5, AA, size=P)
    freqs = rng.uniform(0.0, 2 * np.pi, size=P)
    V = steering_matrix(freqs, P)
    return (V * amps[None, :]) @ V.conj().T + 0.15 * np.eye(P)


def sample_snapshots(T, N, rng):
    """N proper complex Gaussian snapshots with covariance T, one per row."""
    P = T.shape[0]
    L = cholesky(T)
    Z = (rng.standard_normal((N, P)) + 1j * rng.standard_normal((N, P))) / np.sqrt(2)
    return Z @ L.T   # rows x_n = L z_n, E[x x^H] = T


def scm(X):
    """Sample covariance S = (1/N) sum_n x_n x_n^H of snapshots stored as rows."""
    N = X.shape[0]
    return (X.T @ X.conj()) / N
