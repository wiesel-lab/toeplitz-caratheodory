"""Reproduces Figure 1 of the paper: amplitude-only failure on a fixed grid and
the effect of joint optimization, in the population setting (S = C).

The true covariance is the grid example of Section II-C,
    C = b v(w0) v(w0)^H + sigma^2 I,   w0 = pi / K,
the midpoint between the grid frequencies 0 and 2 pi / K, with b = 3 and sigma^2
chosen so that b / (b + sigma^2) = (1 + cos(pi/K)) / 2, strictly above the
threshold cos(pi/K). P = 9 and K in {P, 2P, 3P, 4P}.

Both stages run Algorithm 1: plain softplus amplitudes, the initialization of
Algorithm 1, Armijo backtracking with alpha = 0.3 and beta = 0.5 and no limit
on the number of trial steps (200 is a safety cap that is never reached), and
singular trial points rejected (L = +inf).
  Stage 1: 2,500 iterations, amplitudes only (frequencies fixed on the grid),
           initial amplitude step size 0.03.
  Stage 2: 2,000 iterations from the end of Stage 1, amplitudes and frequencies
           jointly, initial step sizes 1 (amplitudes) and 6e-3 (frequencies).

This script is self-contained: it implements Algorithm 1 directly, exactly as
it was run for the paper.

Usage:
    python experiments/figure1.py                   # figure from results/figure1.json
    python experiments/figure1.py --rerun           # rerun (about a minute)
    python experiments/figure1.py --stage1-eta 1.0  # Stage 1 with the Stage 2 amplitude
                                                    # step size: nearly the same error levels

Outputs: results/figure1.png, results/figure1.json
"""
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

P = 9
LAM = 3.0
K_LIST = [P, 2 * P, 3 * P, 4 * P]
STAGE1_ITERS = 2500
STAGE2_ITERS = 2000
ETA1_A = 0.03                      # Stage 1 initial amplitude step size
ETA2_A, ETA2_W = 1.0, 6e-3         # Stage 2 initial step sizes (amplitudes, frequencies)
ALPHA, BETA = 0.3, 0.5
MAX_BACKTRACK = 200                # safety only: the runs never come close to it

RES_JSON = os.path.join(ROOT, 'results', 'figure1.json')
FIG_PNG = os.path.join(ROOT, 'results', 'figure1.png')


def softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def inv_softplus(y):
    return math.log(math.expm1(y))


def steering(w):
    return np.exp(1j * np.outer(np.arange(P), w))          # P x K matrix of v(w_k)


def covariance(a_raw, w):
    V = steering(w)
    return (V * softplus(a_raw)[None, :]) @ V.conj().T, V


def nll(Chat, S):
    """tr(S Chat^-1) + log det Chat, and +inf when Chat is not positive definite."""
    try:
        L = np.linalg.cholesky(Chat)
    except np.linalg.LinAlgError:
        return np.inf
    logdet = 2.0 * np.sum(np.log(np.real(np.diag(L))))
    X = np.linalg.solve(L, S)
    Y = np.linalg.solve(L, X.conj().T)
    return float(np.real(np.trace(Y)) + logdet)


def gradients(a_raw, w, S):
    Chat, V = covariance(a_raw, w)
    Ci = np.linalg.inv(Chat)
    E = Ci @ (Chat - S) @ Ci
    EV = E @ V
    p = np.real(np.sum(V.conj() * EV, axis=0))                       # v_k^H E v_k
    D = np.arange(P)[:, None]
    dp = -2.0 * np.imag(np.sum(V.conj() * (E @ (D * V)), axis=0))    # p'(w_k) = -2 Im(v^H E D v)
    g_a = sigmoid(a_raw) * p
    g_w = softplus(a_raw) * dp
    return g_a, g_w, nll(Chat, S), Chat


def armijo_step(a_raw, w, S, eta_a, eta_w, joint):
    g_a, g_w, loss, Chat = gradients(a_raw, w, S)
    if not joint:
        g_w = np.zeros_like(g_w)
    sq = eta_a * float(g_a @ g_a) + eta_w * float(g_w @ g_w)
    for n_bt in range(MAX_BACKTRACK + 1):
        a_new, w_new = a_raw - eta_a * g_a, w - eta_w * g_w
        if nll(covariance(a_new, w_new)[0], S) <= loss - ALPHA * sq:
            return a_new, w_new, n_bt
        eta_a, eta_w, sq = BETA * eta_a, BETA * eta_w, BETA * sq
    raise RuntimeError('backtracking limit reached')


def midpoint_covariance(K):
    w0 = math.pi / K
    ratio = 0.5 * (1.0 + math.cos(math.pi / K))
    sigma2 = LAM * (1.0 - ratio) / ratio
    v = steering(np.array([w0]))
    return sigma2 * np.eye(P) + LAM * (v @ v.conj().T), sigma2, ratio


def frob_mse(Chat, C):
    return float(np.mean(np.abs(Chat - C) ** 2))


def run(K, eta1_a=ETA1_A):
    C, sigma2, ratio = midpoint_covariance(K)
    S = C                                                    # population setting
    w = 2 * np.pi * np.arange(K) / K
    a_raw = np.full(K, inv_softplus(np.trace(S).real / (P * K)))
    curve, max_bt = [], 0
    for it in range(STAGE1_ITERS + STAGE2_ITERS):
        curve.append(frob_mse(covariance(a_raw, w)[0], C))
        joint = it >= STAGE1_ITERS
        eta_a, eta_w = (ETA2_A, ETA2_W) if joint else (eta1_a, 0.0)
        a_raw, w, n_bt = armijo_step(a_raw, w, S, eta_a, eta_w, joint=joint)
        max_bt = max(max_bt, n_bt)
    curve.append(frob_mse(covariance(a_raw, w)[0], C))
    return dict(K=K, sigma2=sigma2, ratio=ratio, threshold=math.cos(math.pi / K),
                cond=float(np.linalg.cond(C)), max_backtracks=max_bt, curve=curve)


def make_figure(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'xtick.labelsize': 12,
                         'ytick.labelsize': 12, 'legend.fontsize': 12})
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    cmap = plt.get_cmap('viridis')
    for i, r in enumerate(results):
        mult = r['K'] // P
        lbl = r'$K=P$' if mult == 1 else rf'$K={mult}P$'
        ax.plot(np.arange(len(r['curve'])), r['curve'], color=cmap(i / (len(results) - 1)), lw=2.2, label=lbl)
    ax.axvline(STAGE1_ITERS, color='black', linestyle='--', linewidth=1.2)
    ax.set_yscale('log')
    ax.set_ylim(1e-3, 20)
    ax.text(STAGE1_ITERS - 150, 4.5, 'joint stage', fontsize=11, rotation=90, va='top', ha='right')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Average Frobenius MSE')
    ax.set_title(f'Frobenius MSE vs Iteration, P={P}, Above threshold')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_PNG}')


def summary(results):
    for r in results:
        c = np.array(r['curve'])
        s2 = c[STAGE1_ITERS:]
        cross = int(np.argmax(s2 < 1e-3)) if (s2 < 1e-3).any() else None
        print(f"K={r['K']:2d}: ratio {r['ratio']:.4f} > cos(pi/K) {r['threshold']:.4f}, cond(C)={r['cond']:.0f}, "
              f"stage-1 end {c[STAGE1_ITERS]:.3g}, stage-2 iterations to 1e-3: {cross}, final {c[-1]:.2e}, "
              f"max backtracks {r['max_backtracks']}")


def main():
    args = sys.argv[1:]
    if '--stage1-eta' in args:
        eta = float(args[args.index('--stage1-eta') + 1])
        saved = json.load(open(RES_JSON))['results'] if os.path.exists(RES_JSON) else None
        print(f'Stage 1 error after {STAGE1_ITERS} iterations, initial amplitude step size {eta} '
              f'(in parentheses: {ETA1_A}, as in Figure 1):')
        for i, K in enumerate(K_LIST):
            c = run(K, eta1_a=eta)['curve'][STAGE1_ITERS]
            ref = f"  ({saved[i]['curve'][STAGE1_ITERS]:.3g})" if saved else ''
            print(f'  K={K:2d}: {c:.3g}{ref}')
        return
    if os.path.exists(RES_JSON) and '--rerun' not in args:
        results = json.load(open(RES_JSON))['results']
        print(f'Loaded {RES_JSON} (use --rerun to recompute)')
    else:
        os.makedirs(os.path.dirname(RES_JSON), exist_ok=True)
        results = [run(K) for K in K_LIST]
        json.dump(dict(P=P, lam=LAM, stage1_iters=STAGE1_ITERS, stage2_iters=STAGE2_ITERS, eta1_a=ETA1_A,
                       eta2_a=ETA2_A, eta2_w=ETA2_W, alpha=ALPHA, beta=BETA, results=results), open(RES_JSON, 'w'))
        print(f'Saved {RES_JSON}')
    summary(results)
    make_figure(results)


if __name__ == '__main__':
    main()
