"""Prints every number quoted in the text of Section V of the paper, and the
remark of Section IV that d > 1 in all finite-sample data sets, from the saved
results of figure1.py, figure2.py and figure3.py. Runs in a few seconds.

Usage:
    python experiments/section5_numbers.py
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data import random_covariance, sample_snapshots, scm


def load(name):
    return json.load(open(os.path.join(ROOT, 'results', name)))


def figure1():
    print('Section V-A (Figure 1)')
    d = load('figure1.json')
    n1 = d['stage1_iters']
    for r in d['results']:
        c = np.array(r['curve'])
        s2 = c[n1:]
        cross = int(np.argmax(s2 < 1e-3)) if (s2 < 1e-3).any() else None
        print(f"  K={r['K']:2d}: error {c[n1]:.3g} at the end of Stage 1, below 1e-3 after {cross} "
              f"Stage 2 iterations, final {c[-1]:.1e} (a drop of {np.log10(c[n1] / c[-1]):.1f} orders of magnitude)")
    print('  (Stage 1 with the larger step size of Stage 2: python experiments/figure1.py --stage1-eta 1.0)')


def figure2():
    print('Section V-B (Figure 2)')
    d = load('figure2.json')
    PS, KF, seeds = [15, 20], [1, 1.25, 1.5, 2, 3, 4], range(1, 51)

    def iters(kf):
        return np.array([d[f'P={P}|kf={kf}|seed={s}']['iters'] or np.inf for P in PS for s in seeds], float)

    n_ok = sum(np.isfinite(iters(kf)).sum() for kf in KF)
    print(f'  successful runs: {n_ok} of {len(KF) * len(PS) * len(seeds)}')
    a, b = iters(1), iters(4)
    print(f'  budget 2,500: K=4P succeeds in {(b <= 2500).mean() * 100:.0f}%, K=P in {(a <= 2500).mean() * 100:.0f}%')
    print(f'  median iterations: K=P {np.median(a):,.0f}, K=4P {np.median(b):,.0f}')
    print(f'  hardest instance: K=P {a.max():,.0f}, K=4P {b.max():,.0f} iterations')
    order = sorted(seeds, key=lambda s: -d[f'P=15|kf=1|seed={s}']['iters'])
    h = [d[f'P=15|kf={kf}|seed=4']['iters'] for kf in KF]
    print(f'  panel (b) instance (P=15, seed 4) is number {order.index(4) + 1} from the slowest at K=P; '
          f'iterations by K: {h}, K=P / K=4P = {h[0] / h[-1]:.1f}')
    t = np.array([d[f'P={P}|kf=1|seed={s}']['time'] / d[f'P={P}|kf=4|seed={s}']['time'] for P in PS for s in seeds])
    print(f'  run time K=P / K=4P: shorter for K=4P on {(t > 1).sum()} of {len(t)} covariances, median ratio {np.median(t):.1f}')


def figure3():
    print('Section V-C (Figure 3 and Table 1)')
    d = load('figure3.json')
    rs = list(d.values())
    print(f'  data sets: {len(rs)}')
    print(f'  reference initializations agree within 1e-8 in NLL on {sum(r["mle"]["init_gap"] < 1e-8 for r in rs)} '
          f'of {len(rs)} data sets')
    for r in rs:
        if r['mle']['init_gap'] >= 1e-8:
            print(f"    disagree: P={r['P']}, M={r['M']}, seed {r['seed']}, gap {r['mle']['init_gap']:.1e}")
    low = min(g['final']['loss'] - r['mle']['loss'] for r in rs for g in r['gd'].values())
    print(f'  lowest NLL of gradient descent minus the reference: {low:.1e}')
    for P in [15, 30]:
        sub = [r for r in rs if r['P'] == P and r['M'] == 2 * P]
        for kf in ['1', '2']:
            cks = sorted(int(c) for c in sub[0]['gd'][kf]['checkpoints'])
            med = {c: np.median([r['gd'][kf]['checkpoints'][str(c)]['loss'] - r['mle']['loss'] for r in sub]) for c in cks}
            first = next((c for c in cks if med[c] <= 1e-7), None)
            capped = np.mean([r['gd'][kf]['final']['iters'] >= 30000 for r in sub]) * 100
            print(f'  P={P}, M=2P, K={"P" if kf == "1" else "2P"}: median gap first within 1e-7 at budget {first}, '
                  f'final median gap {med[cks[-1]]:.1e}, capped {capped:.0f}%')
    for P in [15, 30]:
        sub = [r for r in rs if r['P'] == P]
        for kf in ['1', '2']:
            frac = np.mean([r['gd'][kf]['final']['loss'] - r['mle']['loss'] < 1e-6 for r in sub]) * 100
            print(f'  all sample sizes, P={P}, K={"P" if kf == "1" else "2P"}: final gap below 1e-6 in {frac:.0f}% of the runs')


def section4_d():
    print('Section IV (d of Theorem 2 in the finite-sample data sets)')
    ds = []
    for P in [15, 30]:
        for mf in [1, 2, 4, 8]:
            for seed in range(40):
                rng = np.random.default_rng(np.random.SeedSequence([P, mf, seed]))
                C = random_covariance(P, rng)
                X = sample_snapshots(C, mf * P, rng)
                S = scm(X)
                S = 0.5 * (S + S.conj().T)
                lam, U = np.linalg.eigh(C)
                Ci = (U / np.sqrt(lam)) @ U.conj().T
                ds.append(np.linalg.norm(Ci @ (S - C) @ Ci))
    print(f'  d over the {len(ds)} data sets: smallest {min(ds):.2f}, largest {max(ds):.2f}')


if __name__ == '__main__':
    figure1()
    figure2()
    figure3()
    section4_d()
