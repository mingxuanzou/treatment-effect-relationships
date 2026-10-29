#!/usr/bin/env python3
"""Three finite-population figures, fixed kappa=2 and target X, with W0=(X,U,X-by-U).

Only reads completed simulation summaries. No fitting is done; display selection includes OLS, model assistance with W0, and Ding.
Run: python3 presentation/plot_results.py [--input results/summary.csv --output figures]
"""
from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['V11', 'V22', 'V12', 'rho', 'theta', 'alpha']
LABELS = [r'$V_{11}$', r'$V_{22}$', r'$V_{12}$', r'$\rho$', r'$\theta$', r'$\alpha$']
METHODS = ['ols', 'model_assisted', 'ding_plugin']
STYLE = {
    'ols': dict(color='#2675AA', marker='o', ls='-', label='OLS'),
    'model_assisted': dict(color='#D55E00', marker='s', ls='-', label=r'Model assisted ($W_0$)'),
    'ding_plugin': dict(color='#777777', marker='D', ls='-', label='Ding'),
}
METHOD_LABEL = {m: STYLE[m]['label'] for m in METHODS}
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.3,
    'axes.titlesize': 12.5, 'axes.labelsize': 10.5,
    'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#B0B4BA', 'axes.linewidth': .7,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})


def fmt_x(value, _):
    return f'{value / 1000:g}k' if value >= 1000 else f'{int(value)}'


def draw(data, metric, output, save_pdf=True):
    metric_title = {'coverage': '95% interval coverage', 'bias': 'Bias', 'rmse': 'RMSE'}[metric]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    fig.subplots_adjust(left=.075, right=.985, bottom=.10, top=.795, wspace=.25, hspace=.43)
    fig.suptitle(f'Finite population: {metric_title}', fontsize=17.5, fontweight='bold',
                 x=.075, y=.975, ha='left')
    handles = [Line2D([0], [0], lw=1.9, markersize=4.5, **STYLE[m]) for m in METHODS]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.067, .895), frameon=False,
               ncol=3, columnspacing=2.0, handlelength=2.7, fontsize=10.3)
    ngrid = sorted(data.n.unique())
    y_min = .90
    for ax, target, label in zip(axes.flat, TARGETS, LABELS):
        d = data[data.estimand == target]
        truth = float(d.truth.iloc[0])
        assert np.allclose(d.truth, truth)
        ax.set_title(label, pad=9)
        if metric == 'coverage':
            ax.axhline(.95, color='#A0A5AC', lw=.85, ls='-', zorder=1)
        elif metric == 'bias':
            ax.axhline(0, color='#A0A5AC', lw=.9, ls='-', zorder=1)
        for method in METHODS:
            q = d[d.method == method].sort_values('n')
            assert q.n.tolist() == ngrid
            st = STYLE[method]
            # Each point is at its actual n; no artificial horizontal displacement.
            ax.plot(q.n, q[metric],
                    color=st['color'], marker=st['marker'], ls=st['ls'],
                    lw=1.9, markersize=4.5, markeredgewidth=.7,
                    zorder=3, markerfacecolor=st['color'])
        ax.set_xscale('log')
        ax.set_xticks(ngrid)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='minor', bottom=False)
        ax.set_xlim(ngrid[0] / 1.17, ngrid[-1] * 1.17)
        ax.grid(axis='y', color='#E8EBEF', lw=.6, zorder=0)
        ax.set_axisbelow(True)
        if metric == 'coverage':
            ax.set_ylim(y_min, 1.012)
            ax.set_yticks(np.arange(y_min, 1.001, .05))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0%}'))
        elif metric == 'bias':
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        else:
            assert np.all(d[metric] - 2 * d[metric + '_mcse'] > 0)
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
        if ax in axes[1]:
            ax.set_xlabel('Population size n (log scale)')
    for ax in axes[:, 0]:
        ax.set_ylabel({'coverage': 'Coverage', 'bias': 'Mean estimate − truth',
                       'rmse': 'RMSE (log scale)'}[metric])
    name = output / f'finite_population_{metric}'
    if save_pdf:
        buffer = BytesIO()
        fig.savefig(buffer, format='pdf')
        payload = buffer.getvalue()
        assert payload.rstrip().endswith(b'%%EOF')
        temporary = name.with_suffix('.pdf.tmp')
        temporary.write_bytes(payload)
        temporary.replace(name.with_suffix('.pdf'))
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png', dpi=180)
    png_temporary = name.with_suffix('.png.tmp')
    png_temporary.write_bytes(png_buffer.getvalue())
    png_temporary.replace(name.with_suffix('.png'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'results' / 'summary.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    data = data[data.estimand.isin(TARGETS) & data.method.isin(METHODS)].copy()
    assert len(data) == 90
    assert set(data.method) == set(METHODS)
    assert set(data.n) == {200, 500, 1000, 2000, 5000}
    assert set(data.kappa) == {2}
    assert set(data.p) == {.5}
    assert set(data.rho) == {.5}
    assert set(data.xset) == {'large'}
    assert set(data.reps) == {2000}
    assert set(data.estimate_success) == {2000}
    assert set(data.interval_success) == {2000}
    assert not data.duplicated(['n', 'method', 'estimand']).any()
    assert np.isfinite(data[['coverage', 'bias', 'rmse', 'coverage_mcse', 'bias_mcse', 'rmse_mcse']]).all().all()
    args.output.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output / 'finite_population_plot_data.csv', index=False)
    for metric in ['coverage', 'bias', 'rmse']:
        draw(data, metric, args.output)
    manifest = {
        'source': str(args.input), 'source_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
        'selection': 'Fixed kappa=2, large X, p=.5, rho=.5; OLS, model-assisted W0=(X,U,X-by-U), and Ding; all five sizes.',
        'source_rows': len(data), 'assignments_per_population': 2000,
        'methods': {m: METHOD_LABEL[m] for m in METHODS},
        'uncertainty': 'Error bars and shaded bands are omitted at user request; Monte Carlo SEs remain in source summaries and paired contrasts. Coverage plots retain the 95% reference line.',
        'coverage_event': 'Untruncated separate 95% Wald interval contains exact fixed-population target; all attempted assignments in denominator.',
        'missing_estimates': int((data.reps - data.estimate_success).sum()),
        'missing_intervals': int((data.reps - data.interval_success).sum()),
    }
    (args.output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
