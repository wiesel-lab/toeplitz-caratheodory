"""Carathéodory parameterization of a Toeplitz covariance and Algorithm 1 of the paper.

The covariance is parameterized by K atoms with softplus amplitudes,

    C-hat(a, w) = sum_k s(a_k) v(w_k) v(w_k)^H,   s(a) = log(1 + e^a),
    v(w) = [1, e^{iw}, ..., e^{iw(P-1)}]^T,

and Algorithm 1 minimizes the per-sample Gaussian negative log-likelihood

    L(a, w) = tr(S C-hat^{-1}) + log|C-hat|

by gradient descent with separate step sizes for the amplitudes and the
frequencies, and an Armijo backtracking line search that reduces both step
sizes by the same factor. Trial steps at which C-hat is singular get L = +inf
and are rejected.

The building blocks below (build_cov, nll, gradients, armijo_step) are the
exact code used for Figures 2 and 3 and Table 1. `algorithm1` runs Algorithm 1
as stated in the paper.
"""

import numpy as np


def softplus(x):
    return np.logaddexp(0.0, x)


def inv_softplus(y):
    return np.log(np.expm1(y))


def sigmoid(x):
    """Derivative of the softplus function."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def steering_matrix(freqs, P):
    """P x K matrix whose columns are the steering vectors v(w_k)."""
    return np.exp(1j * np.arange(P)[:, None] * freqs[None, :])


def build_cov(freqs, weights, P, eps=0.0):
    """C-hat = sum_k s(weights_k) v(freqs_k) v(freqs_k)^H + eps I.

    Returns C-hat, the steering matrix V and the amplitudes s(weights).
    All experiments in the paper use eps = 0.
    """
    V = steering_matrix(freqs, P)
    s = softplus(weights)
    return (V * s[None, :]) @ V.conj().T + eps * np.eye(P), V, s


def nll(C, S):
    """tr(S C^{-1}) + log|C|, or +inf when C is not positive definite."""
    sign, ld = np.linalg.slogdet(C)
    if sign.real <= 0 or not np.isfinite(ld):
        return np.inf
    val = float(np.trace(np.linalg.solve(C, S)).real) + float(ld)
    return val if np.isfinite(val) else np.inf


def gradients(C, S, V, s, weights, P):
    """Gradients of the NLL with respect to the raw amplitudes and the frequencies.

    With E = C^{-1} (C - S) C^{-1} and D = diag(0, 1, ..., P-1),
        dL/da_k = s'(a_k) v_k^H E v_k,   dL/dw_k = -2 s(a_k) Im(v_k^H E D v_k).
    """
    Cinv = np.linalg.inv(C)
    E = Cinv @ (C - S) @ Cinv
    gw = sigmoid(weights) * np.einsum('pk,pk->k', V.conj(), E @ V).real
    gf = -2.0 * s * np.einsum('pk,pk->k', V.conj(), E @ (np.arange(P)[:, None] * V)).imag
    return gw, gf


def armijo_step(freqs, weights, S, gf, gw, loss, P, eta_w, eta_a, alpha, beta,
                max_trials, eps=0.0):
    """Armijo backtracking with the fixed step-size matrix diag(eta_a I, eta_w I).

    Tries the steps q = 1, beta, beta^2, ... (at most `max_trials` of them, or
    without a limit if max_trials is None) and accepts the first one with
        L(new) <= L(old) - alpha q (eta_a ||g_a||^2 + eta_w ||g_w||^2).
    Returns (freqs, weights, True) for the accepted step, or the unchanged
    parameters and False if no trial step was accepted.
    """
    lr_f, lr_a = eta_w, eta_a
    trial = 0
    while max_trials is None or trial < max_trials:
        cf, cw = freqs - lr_f * gf, weights - lr_a * gw
        nl = nll(build_cov(cf, cw, P, eps)[0], S)
        if np.isfinite(nl) and nl <= loss - alpha * (lr_f * float(gf @ gf) + lr_a * float(gw @ gw)):
            return cf, cw, True
        lr_f *= beta
        lr_a *= beta
        trial += 1
    return freqs, weights, False


def initialize(S, K):
    """Initialization of Algorithm 1: frequencies on the uniform grid and equal
    amplitudes s(a_k) = tr(S) / (P K), so that C-hat^(0) = (tr(S) / P) I."""
    P = S.shape[0]
    gamma = float(np.trace(S).real) / P
    freqs = np.arange(K, dtype=float) / K * 2 * np.pi
    weights = np.full(K, float(inv_softplus(gamma / K)))
    return freqs, weights


def algorithm1(S, K, T, eta_a=0.3, eta_w=6e-3, alpha=0.3, beta=0.5):
    """Algorithm 1 of the paper: T iterations of gradient descent on L(a, w).

    Args:
        S: P x P sample covariance with tr(S) > 0.
        K: number of atoms, K >= P.
        T: number of iterations.
        eta_a, eta_w: initial step sizes of the amplitudes and the frequencies.
        alpha, beta: backtracking parameters, alpha in (0, 0.5), beta in (0, 1).

    Returns:
        The covariance estimate C-hat (P x P) and the final frequencies and raw
        amplitudes.

    The line search has no limit on the number of trial steps, as in the paper.
    It always ends, because a small enough step satisfies the Armijo condition.
    """
    S = np.asarray(S, dtype=complex)
    P = S.shape[0]
    if not float(np.trace(S).real) > 0:
        raise ValueError('Algorithm 1 needs tr(S) > 0')
    if K < P:
        raise ValueError('Algorithm 1 needs K >= P')
    freqs, weights = initialize(S, K)
    for _ in range(T):
        C, V, s = build_cov(freqs, weights, P)
        loss = nll(C, S)
        gw, gf = gradients(C, S, V, s, weights, P)
        freqs, weights, _ = armijo_step(freqs, weights, S, gf, gw, loss, P, eta_w, eta_a,
                                        alpha, beta, max_trials=None)
    return build_cov(freqs, weights, P)[0], freqs, weights
