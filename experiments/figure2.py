"""Reproduces Figure 2 of the paper: convergence of joint gradient descent versus
overparameterization in the population setting (S = C).

For P in {15, 20}, 50 random full-rank Toeplitz covariances each (seeds 1-50,
see src/data.py), and K = round(F P) with F in {1, 1.25, 1.5, 2, 3, 4}
(K = 19 and 22 for P = 15), gradient descent starts from the initialization of
Algorithm 1 and uses the initial step sizes 0.3 (amplitudes) and 6e-3
(frequencies), with alpha = 0.3 and beta = 0.5. A run succeeds once the relative
Frobenius error ||C-hat - C||_F / ||C||_F is below 1e-2. It stops once this error
is below 5e-3, or after 80,000 iterations. The line search tries at most 25
steps per iteration. In the 600 runs of the paper this limit was never reached.

Panel (b) shows the second slowest of the 50 instances with P = 15 at K = P
(seed 4).

Usage:
    python experiments/figure2.py           # figure from results/figure2.json
    python experiments/figure2.py --rerun   # rerun the 600 runs (a few minutes)

Outputs: results/figure2.png, results/figure2.json
"""
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.caratheodory import armijo_step, build_cov, gradients, initialize, nll
from src.data import random_covariance

P_LIST = [15, 20]
KFAC = [1, 1.25, 1.5, 2, 3, 4]          # K = round(kf * P)
N_SEEDS = 50
MAX_ITER = 80000
THR = 0.01                              # success = 1% relative Frobenius error
HARD_P = 15                             # instance shown in panel (b)
HARD_SEED = 4
ALPHA, BETA = 0.3, 0.5
LR_A0, LR_F0 = 3e-1, 6e-3
MAX_TRIALS = 25

RES_JSON = os.path.join(ROOT, 'results', 'figure2.json')
FIG_PNG = os.path.join(ROOT, 'results', 'figure2.png')


def rel_frob(C, S):
    return float(np.linalg.norm(C - S) / np.linalg.norm(S))


def run_trial(job, log_every=50):
    P, kf, seed = job
    rng = np.random.default_rng(seed)
    S = random_covariance(P, rng)          # population setting: S = C
    K = int(round(kf * P))
    freqs, weights = initialize(S, K)

    C0 = build_cov(freqs, weights, P)[0]
    traj, iter_thr = [], None
    t0 = time.time()
    for i in range(MAX_ITER):
        C, V, s = build_cov(freqs, weights, P)
        loss = nll(C, S)
        gw, gf = gradients(C, S, V, s, weights, P)
        gn = math.sqrt(float(gw @ gw) + float(gf @ gf))
        err = rel_frob(C, S)
        if i % log_every == 0:
            traj.append((i, err))
        if iter_thr is None and err <= THR:
            iter_thr = i
            traj.append((i, err))
        if (iter_thr is not None and err <= THR * 0.5) or gn < 1e-10:
            traj.append((i, err))
            break
        freqs, weights, ok = armijo_step(freqs, weights, S, gf, gw, loss, P, LR_F0, LR_A0,
                                         ALPHA, BETA, MAX_TRIALS)
        if not ok:
            traj.append((i, err))
            break
    if not traj or traj[-1][0] != i:
        traj.append((i, err))
    return (f"P={P}|kf={kf}|seed={seed}",
            dict(P=P, kf=kf, K=K, seed=seed, iters=iter_thr, final_err=traj[-1][1],
                 init_diag=float(C0[0, 0].real), true_diag=float(S[0, 0].real),
                 time=time.time() - t0, traj=[[int(a), float(b)] for a, b in traj]))


def sweep(rerun=False):
    os.makedirs(os.path.dirname(RES_JSON), exist_ok=True)
    out = {} if rerun or not os.path.exists(RES_JSON) else json.load(open(RES_JSON))
    jobs = [(P, kf, s) for P in P_LIST for kf in KFAC for s in range(1, N_SEEDS + 1)
            if f"P={P}|kf={kf}|seed={s}" not in out]
    if not jobs:
        print(f'Loaded {RES_JSON} (use --rerun to recompute)')
        return out
    jobs.sort(key=lambda j: j[1])           # small K first: K = P runs are the slowest
    print(f'{len(jobs)} runs to do', flush=True)
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        for n, (key, r) in enumerate(pool.imap_unordered(run_trial, jobs), 1):
            out[key] = r
            print(f"[{n}/{len(jobs)}] {key}: iterations to 1% = {r['iters']}  t={r['time']:.0f}s", flush=True)
            if n % 20 == 0 or n == len(jobs):
                json.dump(out, open(RES_JSON, 'w'))
    json.dump(out, open(RES_JSON, 'w'))
    return out


def make_figure(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    seeds = range(1, N_SEEDS + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=len(KFAC) - 1)

    # ---- (a) success rate vs iteration budget ----
    ax = axes[0]
    budgets = np.unique(np.concatenate([np.logspace(2, np.log10(MAX_ITER), 300), [MAX_ITER]]))
    for i, kf in enumerate(KFAC):
        iters = []
        for P in P_LIST:
            for s in seeds:
                r = data[f"P={P}|kf={kf}|seed={s}"]
                iters.append(r['iters'] if r['iters'] is not None else np.inf)
        iters = np.array(iters, dtype=float)
        rate = [(iters <= b).mean() * 100 for b in budgets]
        lbl = '$K=P$' if kf == 1 else f'$K={kf}P$'
        ax.plot(budgets, rate, color=cmap(norm(i)), lw=2.0, label=lbl)
    ax.set_xscale('log')
    ax.set_xlabel('Iteration budget')
    ax.set_ylabel('Success rate (%)')
    ax.set_xlim([100, MAX_ITER])
    ax.set_ylim([-3, 103])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper left')
    ax.text(0.02, 1.02, '(a)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    # ---- (b) hard instance trajectories ----
    ax = axes[1]
    xmax = int(data[f"P={HARD_P}|kf=1|seed={HARD_SEED}"]['iters'] * 1.12)
    for i, kf in enumerate(KFAC):
        tr = np.array(data[f"P={HARD_P}|kf={kf}|seed={HARD_SEED}"]['traj'])
        m = tr[:, 0] <= xmax
        lbl = '$K=P$' if kf == 1 else f'$K={kf}P$'
        ax.plot(tr[m, 0], tr[m, 1], color=cmap(norm(i)), lw=1.9, label=lbl)
    ax.axhline(THR, color='red', ls=':', lw=1.3)
    ax.text(0.27 * xmax, THR * 0.80, '1% threshold',
            ha='center', va='top', fontsize=9, color='red')
    ax.set_yscale('log')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative Frobenius error')
    ax.set_xlim([0, xmax])
    ax.set_ylim([4e-3, 1.3])
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=9, loc='upper right')
    ax.text(0.02, 1.02, '(b)', transform=ax.transAxes, fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_PNG, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIG_PNG}')

    print('\nSuccess rate (%) at selected budgets (pooled over P and seeds):')
    for b in [1000, 2000, 5000, MAX_ITER]:
        cells = []
        for kf in KFAC:
            iters = []
            for P in P_LIST:
                for s in seeds:
                    r = data[f"P={P}|kf={kf}|seed={s}"]
                    iters.append(r['iters'] if r['iters'] is not None else np.inf)
            cells.append(f"K={kf}P:{(np.array(iters) <= b).mean()*100:5.1f}")
        print(f"  budget={b:>6}: " + '  '.join(cells))


if __name__ == '__main__':
    make_figure(sweep(rerun='--rerun' in sys.argv[1:]))
