"""Reference maximum likelihood estimate of a Toeplitz covariance, and the CRB.

The reference MLE of Figure 3 and Table 1 runs Fisher scoring in the lag
coordinates from two initializations and keeps the solution with the lower NLL:

  1. the diagonally averaged sample covariance, made positive definite by
     shrinking it toward c_0 I until its smallest eigenvalue is 1e-4 c_0
     (`pd_repair_shrink`), and
  2. a multisegment Burg estimate of order P-1 (`burg_multisegment`).

Each Fisher-scoring run (`fisher_scoring`) uses an Armijo line search with
c = 1e-4 and at most 40 step halvings, and rejects trial points that are not
positive definite. It stops after 200 iterations, when the Newton decrement
g^T J_1^{-1} g falls below 1e-10 (2P-1), or when no trial step is accepted.

Conventions: theta = [c_0, Re c_1..c_{P-1}, Im c_1..c_{P-1}] (2P-1 real
parameters), T[i, j] = c_{i-j} for i >= j and T[i, j] = conj(c_{j-i}) for i < j.
"""

import numpy as np
from numpy.linalg import cholesky, LinAlgError


def lags_to_theta(c):
    """c: complex (P,) first-column lags -> theta real (2P-1,)."""
    c = np.asarray(c)
    return np.concatenate([[c[0].real], c[1:].real, c[1:].imag])


def theta_to_lags(theta):
    P = (len(theta) + 1) // 2
    c = np.empty(P, dtype=np.complex128)
    c[0] = theta[0]
    c[1:] = theta[1:P] + 1j * theta[P:]
    return c


def toeplitz_from_lags(c):
    """Hermitian Toeplitz matrix with first column c."""
    P = len(c)
    idx = np.subtract.outer(np.arange(P), np.arange(P))  # i - j
    full = np.concatenate([np.conj(c[:0:-1]), c])         # index k -> full[k+P-1]
    return full[idx + P - 1]


def deriv_stack(P):
    """Stack of dT/dtheta_i, shape (2P-1, P, P), complex128. Cached per P."""
    d = 2 * P - 1
    D = np.zeros((d, P, P), dtype=np.complex128)
    i, j = np.indices((P, P))
    D[0] = np.eye(P)
    for k in range(1, P):
        Ek = (i - j == k).astype(np.complex128)   # ones on k-th subdiagonal
        D[k] = Ek + Ek.T                          # d/d Re c_k
        D[P - 1 + k] = 1j * Ek - 1j * Ek.T        # d/d Im c_k
    return D


_DERIV_CACHE = {}


def get_derivs(P):
    if P not in _DERIV_CACHE:
        _DERIV_CACHE[P] = deriv_stack(P)
    return _DERIV_CACHE[P]


def fisher_information(T):
    """Per-snapshot FIM I_1, shape (d, d) real. I_ij = tr(T^-1 D_i T^-1 D_j)."""
    P = T.shape[0]
    D = get_derivs(P)
    Tinv = np.linalg.inv(T)
    A = np.einsum('ab,ibc->iac', Tinv, D)          # T^-1 D_i
    return np.real(np.einsum('iab,jba->ij', A, A))


def diag_averaging(S):
    """First-column lags by averaging the diagonals of S."""
    P = S.shape[0]
    c = np.empty(P, dtype=np.complex128)
    for k in range(P):
        c[k] = np.mean(np.diagonal(S, offset=-k))  # S[i+k, i]
    c[0] = c[0].real
    return c


def pd_repair_shrink(c, rel_floor=1e-4):
    """Shrink toward c_0 * I (stays Toeplitz): T_lam = (1-lam) T + lam c0 I.

    Eigenvalues are linear in lam with shared eigenvectors, so the minimal
    lam achieving min-eig >= rel_floor * c0 is closed form.
    """
    T = toeplitz_from_lags(c)
    mu_min = float(np.linalg.eigvalsh(T)[0])
    c0 = float(c[0].real)
    floor = rel_floor * c0
    if mu_min >= floor:
        return c.copy(), 0.0
    lam = (floor - mu_min) / (c0 - mu_min)
    lam = min(max(lam, 0.0), 1.0)
    c_rep = (1 - lam) * c
    c_rep[0] = (1 - lam) * c[0] + lam * c0
    return c_rep, lam


def burg_multisegment(X, order=None):
    """Complex Burg jointly over segments (snapshots). Returns lags c (P,).

    AR convention: e_t = x_t + sum_j a_j x_{t-j};
    updates f_m = f_{m-1} + k_m b_{m-1}^-, b_m = b_{m-1}^- + conj(k_m) f_{m-1}.
    |k_m| <= 1 by Cauchy-Schwarz -> PD Toeplitz by construction.
    """
    N, P = X.shape
    if order is None:
        order = P - 1
    r0 = float(np.mean(np.abs(X) ** 2))
    f = X.copy()          # forward errors, valid indices order..P-1 handled by slicing
    b = X.copy()
    ks = np.zeros(order, dtype=np.complex128)
    for m in range(1, order + 1):
        fm = f[:, m:]          # f_{m-1}[t], t = m..P-1
        bm = b[:, m - 1:-1]    # b_{m-1}[t-1]
        num = -2.0 * np.sum(fm * np.conj(bm))
        den = np.sum(np.abs(fm) ** 2 + np.abs(bm) ** 2)
        k = num / den if den > 0 else 0.0
        ks[m - 1] = k
        f_new = fm + k * bm
        b_new = bm + np.conj(k) * fm
        f = np.concatenate([np.zeros((N, m), dtype=X.dtype), f_new], axis=1)
        b = np.concatenate([np.zeros((N, m), dtype=X.dtype), b_new], axis=1)
    return lags_from_reflection(r0, ks, P)


def lags_from_reflection(r0, ks, P):
    """Inverse Levinson: reflection coefficients + r0 -> lags r_0..r_{P-1}."""
    r = np.zeros(P, dtype=np.complex128)
    r[0] = r0
    a = np.zeros(P, dtype=np.complex128)  # a[1..m] current AR coefficients
    E = r0
    for m in range(1, len(ks) + 1):
        k = ks[m - 1]
        # k_m = -(r_m + sum_{j=1}^{m-1} a_j r_{m-j}) / E_{m-1}
        acc = np.dot(a[1:m], r[m - 1:0:-1]) if m > 1 else 0.0
        r[m] = -k * E - acc
        a_new = a.copy()
        a_new[m] = k
        if m > 1:
            a_new[1:m] = a[1:m] + k * np.conj(a[m - 1:0:-1])
        a = a_new
        E = E * (1 - np.abs(k) ** 2)
    return r


def nll_grad_fim(theta, S):
    """Returns (nll, grad, FIM) at theta. FIM is per-snapshot (Slepian-Bangs)."""
    P = S.shape[0]
    c = theta_to_lags(theta)
    T = toeplitz_from_lags(c)
    try:
        L = cholesky(T)
    except LinAlgError:
        return np.inf, None, None
    D = get_derivs(P)
    Tinv = np.linalg.inv(T)
    val = float(2 * np.sum(np.log(np.real(np.diag(L)))) + np.real(np.trace(Tinv @ S)))
    M = Tinv - Tinv @ S @ Tinv
    grad = np.real(np.einsum('ab,iba->i', M, D))
    A = np.einsum('ab,ibc->iac', Tinv, D)
    FIM = np.real(np.einsum('iab,jba->ij', A, A))
    return val, grad, FIM


def fisher_scoring(theta0, S, max_iter=200, tol=1e-10, ridge=1e-10):
    """Fisher-scoring MLE with backtracking (PD + Armijo safeguard).

    Returns (theta, nll, n_iter, converged).
    """
    theta = theta0.copy()
    val, grad, FIM = nll_grad_fim(theta, S)
    if not np.isfinite(val):
        return theta, np.inf, 0, False
    d = len(theta)
    for it in range(max_iter):
        try:
            p = np.linalg.solve(FIM + ridge * np.eye(d), -grad)
        except LinAlgError:
            return theta, val, it, False
        decrement = float(-grad @ p)
        if decrement < tol * d:
            return theta, val, it, True
        alpha, ok = 1.0, False
        for _ in range(40):
            theta_new = theta + alpha * p
            val_new, grad_new, FIM_new = nll_grad_fim(theta_new, S)
            if np.isfinite(val_new) and val_new <= val - 1e-4 * alpha * decrement:
                ok = True
                break
            alpha *= 0.5
        if not ok:
            return theta, val, it, True  # cannot improve -> treat as converged
        theta, val, grad, FIM = theta_new, val_new, grad_new, FIM_new
    return theta, val, max_iter, False


def mle_oracle(S, X, max_iter=200):
    """MLE via Fisher scoring from two independent inits (DA-repaired, Burg).

    Returns dict with theta, nll, iters, converged flag and the NLL gap
    between the two solutions (solver-disagreement diagnostic, outcome D).
    """
    c_da = diag_averaging(S)
    c_da, _ = pd_repair_shrink(c_da)
    c_bg = burg_multisegment(X)
    th1, v1, it1, cv1 = fisher_scoring(lags_to_theta(c_da), S, max_iter=max_iter)
    th2, v2, it2, cv2 = fisher_scoring(lags_to_theta(c_bg), S, max_iter=max_iter)
    gap = abs(v1 - v2)
    if v1 <= v2:
        return dict(theta=th1, nll=v1, iters=it1 + it2, converged=cv1, init_gap=gap)
    return dict(theta=th2, nll=v2, iters=it1 + it2, converged=cv2, init_gap=gap)


def crb_first_row(C, M):
    """CRB on the MSE of the first row of C (average over its P complex entries),
    tr(J^{-1}) / P with J = M J_1 the Fisher information of M snapshots."""
    P = C.shape[0]
    Iinv = np.linalg.inv(fisher_information(C)) / M
    d = np.diag(Iinv)
    # theta = [Re c0, Re c1..c_{P-1}, Im c1..c_{P-1}]
    return float((d[0] + d[1:P].sum() + d[P:].sum()) / P)
