"""Reproduces Table 1 of the paper: per-K diagnostics at the end of the
finite-sample runs of Figure 3, for M = 2P over 40 data sets for each P.

Iterations, time, L - L_MLE and ||grad L|| are medians over the data sets,
"Capped" is the fraction of runs that reached the cap of 30,000 iterations,
and MSE/CRB is the mean. The time of the reference MLE includes both of its
initializations. Run times depend on the machine. The paper reports times on
an Apple M3 Pro laptop with 10 runs in parallel.

Usage:
    python experiments/table1.py    # needs results/figure3.json (see figure3.py)

Prints the rows of Table 1 in LaTeX.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_JSON = os.path.join(ROOT, 'results', 'figure3.json')
MF = 2                                # M = 2P
KF = ['1', '1.1', '1.3', '1.5', '1.7', '2']


def label(kf):
    return '$K=P$' if kf == '1' else f'$K=\\lceil {kf}P\\rceil$'


def main():
    data = json.load(open(RES_JSON))
    PS = sorted({r['P'] for r in data.values()})
    print('% Iterations & Capped & Time [s] & L - L_MLE & ||grad L|| & MSE/CRB')
    for P in PS:
        rs = [r for r in data.values() if r['P'] == P and r['M'] == MF * P]
        print(f'% P={P}, M={MF*P}, {len(rs)} data sets')
        for kf in KF:
            f = [r['gd'][kf]['final'] for r in rs]
            it = np.median([x['iters'] for x in f])
            capped = np.mean([x['iters'] >= 30000 for x in f]) * 100
            t = np.median([x['time'] for x in f])
            gap = np.median([(x['loss'] - r['mle']['loss']) for x, r in zip(f, rs)])
            gn = np.median([x['grad'] for x in f])
            mse = np.mean([x['mse'] / r['crb'] for x, r in zip(f, rs)])
            print(f'{label(kf)} & {it:,.0f} & {capped:.0f}\\% & {t:.2f} & {gap:.1e} & {gn:.1e} & {mse:.2f} \\\\')
        mle = [r['mle'] for r in rs]
        print(f'Reference MLE & -- & -- & {np.median([x["time"] for x in mle]):.2f} & 0 & -- & '
              f'{np.mean([x["mse"]/r["crb"] for x, r in zip(mle, rs)]):.2f} \\\\')


if __name__ == '__main__':
    main()
