#!/usr/bin/env python3
"""Publication figures from saved CSVs; does not rerun the experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.font_manager import fontManager

from visual_config import CM, PURPLE, NIGHT_BLUE, GRAY, set_visual_style
from run_sampling_stability import sha256, atomic_json

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sampling_stability/results'
FIGURES = OUT / 'figures'
COLORS = {'L1': PURPLE, 'L2': NIGHT_BLUE, 'None': GRAY}
PRIMARY = [('overall', 'Overall'), ('digit_1', 'Digit 1'), ('digit_9', 'Digit 9')]


def export(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight', pad_inches=.04)
    fig.savefig(FIGURES / f'{name}.svg', bbox_inches='tight', pad_inches=.04)
    fig.savefig(FIGURES / f'{name}.png', dpi=600, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def label(ax, letter, title):
    ax.set_title(title, loc='center')
    ax.text(-.13, 1.07, letter, transform=ax.transAxes, weight='bold', fontsize=10)
    sns.despine(ax=ax)


def main():
    FIGURES.mkdir(exist_ok=True)
    set_visual_style()
    available = {f.name for f in fontManager.ttflist}
    font = 'Atkinson Hyperlegible' if 'Atkinson Hyperlegible' in available else 'DejaVu Sans'
    plt.rcParams.update({'font.family':font, 'font.sans-serif':[font], 'svg.fonttype':'none',
                         'pdf.fonttype':42, 'savefig.facecolor':'white'})
    # Preserve the literal condition label "None"; pandas otherwise treats it as NA.
    convergence = pd.read_csv(OUT/'convergence.csv', keep_default_na=False)
    summary = pd.read_csv(OUT/'convergence_summary.csv', keep_default_na=False)
    uncertainty = pd.read_csv(OUT/'model_uncertainty.csv', keep_default_na=False)
    halves = pd.read_csv(OUT/'independent_halves.csv', keep_default_na=False)
    contrasts = pd.read_csv(OUT/'condition_contrasts.csv', keep_default_na=False)

    fig, axes = plt.subplots(2, 3, figsize=(18*CM, 11*CM), layout='constrained')
    for col, (task, title) in enumerate(PRIMARY):
        for condition, color in COLORS.items():
            subset = convergence.query('condition == @condition and task == @task')
            for _, model in subset.groupby('model'):
                model = model.sort_values('permutations')
                axes[0,col].plot(model.permutations, model.d, color=color, alpha=.25, lw=.5)
                axes[1,col].plot(model.permutations, model.absolute_difference_from_finite_reference,
                                 color=color, alpha=.25, lw=.5)
            grouped = summary.query('condition == @condition and task == @task').sort_values('permutations')
            axes[0,col].plot(grouped.permutations, grouped.mean_d, 'o-', color=color,
                            ms=2.5, label=condition, lw=1.1)
            axes[1,col].plot(grouped.permutations, grouped.median_absolute_difference, 'o-',
                            color=color, ms=2.5, lw=1.1)
        label(axes[0,col], 'ABC'[col], title)
        label(axes[1,col], 'DEF'[col], title)
        for row in range(2):
            axes[row,col].set_xticks([100, 500, 1000])
        axes[1,col].set_xlabel('Permutations')
    axes[0,0].set_ylabel('D')
    axes[1,0].set_ylabel('|D − D at 1,000|')
    axes[0,0].legend(frameon=False, ncol=3, handlelength=1.2, columnspacing=.8)
    export(fig, 'convergence')

    fig, axes = plt.subplots(2, 3, figsize=(18*CM, 12*CM), layout='constrained')
    for col, (task, title) in enumerate(PRIMARY):
        for group_index, (condition, color) in enumerate(COLORS.items()):
            subset = uncertainty.query('condition == @condition and task == @task').sort_values('trial')
            xpos = group_index*11 + np.arange(10)
            # Percentile intervals can be displaced by nonlinear plug-in bias;
            # draw endpoints independently of the point, never clip to include D.
            axes[0,col].vlines(xpos, subset.ci025, subset.ci975, color=color, lw=.75)
            axes[0,col].scatter(xpos, subset.d, color=color, s=9, zorder=3)
            split = halves.query('condition == @condition and task == @task')
            axes[1,col].scatter(split.first_500_d, split.second_500_d,
                                color=color, s=12, alpha=.9, label=condition)
        axes[0,col].set_xticks([4.5,15.5,26.5], ['L1','L2','None'])
        axes[0,col].set_xlabel('Ten fixed models per condition')
        xmin, xmax = axes[1,col].get_xlim()
        ymin, ymax = axes[1,col].get_ylim()
        low, high = min(xmin, ymin), max(xmax, ymax)
        axes[1,col].plot([low,high],[low,high], color='#232324', lw=.7, ls='--', zorder=0)
        axes[1,col].set_xlim(low, high)
        axes[1,col].set_ylim(low, high)
        axes[1,col].set_aspect('equal', adjustable='box')
        axes[1,col].set_xlabel('D: first 500 permutations')
        label(axes[0,col], 'ABC'[col], title)
        label(axes[1,col], 'DEF'[col], title)
    axes[0,0].set_ylabel('D and 95% Monte Carlo interval')
    axes[1,0].set_ylabel('D: independent second 500')
    axes[1,0].legend(frameon=False, loc='upper left', fontsize=7, handletextpad=.2)
    export(fig, 'uncertainty_and_halves')

    fig, axes = plt.subplots(1, 3, figsize=(18*CM, 6.8*CM), layout='constrained')
    names = ['L2 minus L1','None minus L1','None minus L2']
    tasks = ['overall'] + [f'digit_{i}' for i in range(10)]
    for index, (ax, comparison) in enumerate(zip(axes, names)):
        data = contrasts.query('comparison == @comparison').set_index('task').loc[tasks]
        x = np.arange(11)
        ax.axhline(0, color=GRAY, lw=.7, zorder=0)
        ax.vlines(x, data.conditional_mc_ci025, data.conditional_mc_ci975, color=NIGHT_BLUE, lw=.9)
        ax.scatter(x, data.difference_of_model_means, color=NIGHT_BLUE, s=11, zorder=3)
        ax.set_xticks(x, ['All']+[str(i) for i in range(10)])
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Overall / digit score')
        label(ax, 'ABC'[index], comparison.replace('minus','−'))
    axes[0].set_ylabel('Difference in mean D\n95% conditional Monte Carlo interval')
    export(fig, 'condition_comparisons')

    fig, axes = plt.subplots(2, 5, figsize=(18*CM, 9*CM), layout='constrained')
    for digit, ax in enumerate(axes.flat):
        task = f'digit_{digit}'
        for condition, color in COLORS.items():
            for _, data in convergence.query('condition == @condition and task == @task').groupby('model'):
                ax.plot(data.permutations, data.d, color=color, alpha=.23, lw=.4)
            data = summary.query('condition == @condition and task == @task')
            ax.plot(data.permutations, data.mean_d, color=color, lw=1.1, label=condition)
        ax.set_title(f'Digit {digit}')
        ax.set_xticks([100,1000], ['100','1,000'])
        sns.despine(ax=ax)
        if digit >= 5:
            ax.set_xlabel('Permutations')
        if digit % 5 == 0:
            ax.set_ylabel('D')
    axes[0,0].legend(frameon=False, fontsize=6, handlelength=1)
    export(fig, 'all_digit_convergence')
    atomic_json(OUT/'figure_manifest.json', dict(font=font, style_source='MLP/visual_config.py',
        style_sha256=sha256(ROOT/'visual_config.py'), plot_source_sha256=sha256(Path(__file__)),
        exports=['PDF with embedded TrueType fonts','SVG with editable text','600-dpi PNG'],
        colors=COLORS, csv_sha256={p.name:sha256(p) for p in OUT.glob('*.csv')},
        figure_stems=['convergence','uncertainty_and_halves','condition_comparisons','all_digit_convergence']))


if __name__ == '__main__':
    main()
