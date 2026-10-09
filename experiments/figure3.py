"""Reproduces Figure 3 of the paper (the same runs give Table 1): finite-sample
performance of joint gradient descent versus overparameterization.

For P in {15, 30} and M in {P, 2P, 4P, 8P}, 40 random true covariances each
(src/data.py, seed SeedSequence([P, M/P, trial])), M proper complex Gaussian
snapshots give the sample covariance S. On the same S, every K = ceil(F P) with
F in {1, 1.1, 1.3, 1.5, 1.7, 2} runs gradient descent from the initialization of
Algorithm 1, with the initial step sizes 0.3 (amplitudes) and 6e-3 (frequencies),
alpha = 0.3 and beta = 0.5. A run stops after 30,000 iterations, once the
decrease of L is below 1e-12 |L| for 50 consecutive iterations, or when no step
satisfies the Armijo condition within 30 trial steps, in which case it returns
its last accepted iterate. The frequencies are kept in [0, 2 pi).

The reference MLE and the CRB are computed by src/reference_mle.py.

Usage:
    python experiments/figure3.py           # figure from results/figure3.json
    python experiments/figure3.py --rerun   # rerun the 320 data sets (about 50 minutes on 10 cores)

Outputs: results/figure3.png, results/figure3.json (also used by table1.py)
"""
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.caratheodory import armijo_step, build_cov, gradients, initialize, nll
from src.data import random_covariance, sample_snapshots, scm
from src.reference_mle import crb_first_row, mle_oracle, theta_to_lags, toeplitz_from_lags

warnings.filterwarnings('ignore')

P_LIST = [15, 30]
KFAC = [1, 1.1, 1.3, 1.5, 1.7, 2]     # K = ceil(kf * P)
MFAC = [1, 2, 4, 8]                   # M = mf * P
N_TRIALS = 40                         # random true covariances per (P, M)
MAX_ITER = 30000
CHECKPOINTS = [30, 100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000]
ALPHA, BETA = 0.3, 0.5
LR_A0, LR_F0 = 3e-1, 6e-3
MAX_TRIALS = 30
TOL_REL = 1e-12                       # stop when the relative decrease is below TOL_REL ...
PATIENCE = 50                         # ... for PATIENCE consecutive iterations

RES_JSON = os.path.join(ROOT, 'results', 'figure3.json')
FIG_PNG = os.path.join(ROOT, 'results', 'figure3.png')
MF_PLOT = 2                           # M = 2P in all three panels
P_MSE = 30                            # dimension of panel (c)
GAP_FLOOR = 1e-9                      # below this the gap is at the stopping tolerance


def mse_first_row(Chat, C):
    return float(np.mean(np.abs(Chat[0, :] - C[0, :]) ** 2))


def run_gd(S, C, K, P):
    freqs, weights = initialize(S, K)
    ck = {}
    t0 = time.perf_counter()
    stable, it, loss_prev = 0, 0, None
    stopped_at = None
    for it in range(MAX_ITER + 1):
        Cm, V, s = build_cov(freqs, weights, P)
        loss = nll(Cm, S)
        gw, gf = gradients(Cm, S, V, s, weights, P)
        gn = math.sqrt(float(gw @ gw) + float(gf @ gf))
        if it in CHECKPOINTS:
            ck[it] = dict(loss=loss, grad=gn, mse=mse_first_row(Cm, C),
                          time=time.perf_counter() - t0, iters=it)
        if loss_prev is not None and (loss_prev - loss) <= TOL_REL * abs(loss):
            stable += 1
        else:
            stable = 0
        if stable >= PATIENCE or it == MAX_ITER:
            stopped_at = it
            break
        loss_prev = loss
        cf, cw, ok = armijo_step(freqs, weights, S, gf, gw, loss, P, LR_F0, LR_A0,
                                 ALPHA, BETA, MAX_TRIALS)
        if not ok:
            stopped_at = it
            break
        freqs, weights = np.mod(cf, 2 * np.pi), cw
    final = dict(loss=loss, grad=gn, mse=mse_first_row(Cm, C),
                 time=time.perf_counter() - t0, iters=stopped_at)
    # checkpoints after stopping hold the final state
    for c in CHECKPOINTS:
        if c not in ck:
            ck[c] = dict(final, budget_after_stop=True)
    return dict(checkpoints={str(k): v for k, v in ck.items()}, final=final)


def run_job(job):
    P, mf, seed = job
    M = mf * P
    rng = np.random.default_rng(np.random.SeedSequence([P, mf, seed]))
    C = random_covariance(P, rng)
    X = sample_snapshots(C, M, rng)
    S = scm(X)
    S = 0.5 * (S + S.conj().T)

    out = dict(P=P, M=M, seed=seed, crb=crb_first_row(C, M))
    t0 = time.perf_counter()
    mle = mle_oracle(S, X)
    Cmle = toeplitz_from_lags(theta_to_lags(mle['theta']))
    out['mle'] = dict(loss=float(mle['nll']), mse=mse_first_row(Cmle, C),
                      time=time.perf_counter() - t0, converged=bool(mle['converged']),
                      init_gap=float(mle['init_gap']))
    out['gd'] = {}
    for kf in KFAC:
        K = int(math.ceil(kf * P - 1e-9))
        out['gd'][str(kf)] = run_gd(S, C, K, P)
    return f"P={P}|mf={mf}|seed={seed}", out


def sweep(rerun=False):
    os.makedirs(os.path.dirname(RES_JSON), exist_ok=True)
    out = {} if rerun or not os.path.exists(RES_JSON) else json.load(open(RES_JSON))
    jobs = [(P, mf, s) for P in P_LIST for mf in MFAC for s in range(N_TRIALS)
            if f"P={P}|mf={mf}|seed={s}" not in out]
    if not jobs:
        print(f'Loaded {RES_JSON} (use --rerun to recompute)')
        return out
    print(f'{len(jobs)} data sets to run', flush=True)
    jobs.sort(key=lambda j: -j[0])    # P = 30 first: these are the slowest
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for n, (key, r) in enumerate(pool.imap_unordered(run_job, jobs), 1):
            out[key] = r
            g = r['gd']
            print(f"[{n}/{len(jobs)}] {key}: MLE/CRB={r['mle']['mse']/r['crb']:.2f} "
                  + ' '.join(f"K={kf}P:{g[kf]['final']['iters']}it,"
                             f"{g[kf]['final']['mse']/r['crb']:.2f}" for kf in g), flush=True)
            if n % 10 == 0 or n == len(jobs):
                json.dump(out, open(RES_JSON, 'w'))
    json.dump(out, open(RES_JSON, 'w'))
    return out


def make_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    KF = ['1', '1.1', '1.3', '1.5', '1.7', '2']
    PS = sorted({r['P'] for r in data.values()})
    CKS = sorted({int(c) for r in data.values() for c in r['gd']['1']['checkpoints']})
    # viridis without its pale-yellow end, so the largest K stays visible on white
    colors = {kf: plt.cm.viridis(0.05 + 0.78 * i / (len(KF) - 1)) for i, kf in enumerate(KF)}
    plt.rcParams.update({'font.size': 14, 'axes.titlesize': 14.5, 'legend.fontsize': 13.5})

    def label(kf):
        return '$K=P$' if kf == '1' else f'$K=\\lceil {kf}P\\rceil$'

    def runs(P, mf):
        return [r for r in data.values() if r['P'] == P and r['M'] == mf * P]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9))

    # ---- (a,b): likelihood gap vs iteration budget ----
    for j, P in enumerate(PS):
        ax = axes[j]
        rs = runs(P, MF_PLOT)
        for kf in KF:
            med = [np.median([max(r['gd'][kf]['checkpoints'][str(c)]['loss'] - r['mle']['loss'], GAP_FLOOR)
                              for r in rs]) for c in CKS]
            ax.plot(CKS, med, color=colors[kf], lw=2.2, label=label(kf))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(GAP_FLOOR * 0.7, None)
        ax.set_xlabel('Iteration budget')
        ax.set_ylabel(r'$\mathcal{L}-\mathcal{L}_{\mathrm{MLE}}$ (median)')
        ax.set_title(f'({"ab"[j]}) Likelihood gap, $P={P}$, $M={MF_PLOT}P$', loc='left')
        ax.grid(True, which='major', alpha=0.3)

    # ---- (c): MSE/CRB vs iteration budget, P = P_MSE, M = 2P ----
    ax = axes[2]
    rs = runs(P_MSE, MF_PLOT)
    for kf in KF:
        m = [np.mean([r['gd'][kf]['checkpoints'][str(c)]['mse'] / r['crb'] for r in rs]) for c in CKS]
        ax.plot(CKS, m, color=colors[kf], lw=2.2, label=label(kf))
    mm = np.mean([r['mle']['mse'] / r['crb'] for r in rs])
    ax.axhline(mm, color='k', ls='--', lw=1.5, label='Reference MLE')
    ax.axhline(1.0, color='0.45', ls=':', lw=1.3, label='CRB')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(100, max(CKS))
    ax.set_xlabel('Iteration budget')
    ax.set_ylabel('MSE / CRB (mean)')
    ax.set_title(f'(c) Accuracy vs budget, $P={P_MSE}$, $M={MF_PLOT}P$', loc='left')
    ax.grid(True, which='major', alpha=0.3)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(['1', '2', '5', '10'])
    ax.minorticks_off()

    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=8, frameon=False, bbox_to_anchor=(0.5, 1.11),
               columnspacing=1.2, handlelength=1.6)
    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_PNG}')


if __name__ == '__main__':
    make_figure(sweep(rerun='--rerun' in sys.argv[1:]))
