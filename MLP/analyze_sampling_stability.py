#!/usr/bin/env python3
"""Analyze saved permutation vectors; no model inference or resampling images."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sampling_stability.statistics import TASKS, BUDGETS, all_d, whole_permutation_bootstrap
from run_sampling_stability import atomic_json, atomic_npz, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sampling_stability/results'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    args = parser.parse_args()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    if len(manifest['models']) != 30 or any('output_sha256' not in m for m in manifest['models']):
        raise RuntimeError('All 30 models must be complete before analyzing.')
    convergence, uncertainty, halves, contributions, quality, seeds = [], [], [], [], [], []
    bootstrap_models = {condition: [] for condition in ['L1', 'L2', 'None']}
    reference_models = {condition: [] for condition in bootstrap_models}
    source_hash = sha256(ROOT / 'sampling_stability/statistics.py')
    for record in manifest['models']:
        start = time.monotonic()
        key = record['key']
        path = ROOT / record['output']
        assert sha256(path) == record['output_sha256']
        with np.load(path) as saved:
            marginals = saved['marginals']
            permutations = saved['permutations']
            metadata = json.loads(str(saved['metadata']))
            cc, pc = saved['correct_counts'], saved['predicted_counts']
        assert marginals.shape == (1000, 200, 10)
        assert np.all(np.sort(permutations, axis=1) == np.arange(200))
        assert len(np.unique(permutations, axis=0)) == 1000
        condition, trial = record['condition'], record['trial']
        common = dict(condition=condition, trial=trial, model=key)
        mean = marginals.mean(0)
        se = marginals.std(0, ddof=1) / np.sqrt(len(marginals))
        reference = all_d(mean)
        first_half, second_half = all_d(marginals[:500].mean(0)), all_d(marginals[500:].mean(0))
        bootstrap_path = OUT / f'bootstrap_{key}.npz'
        bootstrap_identity = dict(marginals_sha256=record['output_sha256'],
                 statistics_source_sha256=source_hash, resamples=args.bootstrap_resamples,
                 seed=record['bootstrap_seed'], method='whole-row multinomial percentile bootstrap')
        if bootstrap_path.exists():
            with np.load(bootstrap_path) as saved:
                assert json.loads(str(saved['identity'])) == bootstrap_identity
                bootstrap = saved['d']
        else:
            with threadpool_limits(limits=6):
                bootstrap = whole_permutation_bootstrap(marginals, record['bootstrap_seed'], args.bootstrap_resamples)
            atomic_npz(bootstrap_path, d=bootstrap, identity=json.dumps(bootstrap_identity, sort_keys=True))
        lower, upper = np.quantile(bootstrap, [.025, .975], axis=0)
        mc_sd = bootstrap.std(0, ddof=1)
        for task_index, task in enumerate(TASKS):
            uncertainty.append(dict(**common, task=task, permutations=1000,
                d=reference[task_index], ci025=lower[task_index], ci975=upper[task_index],
                interval_width=upper[task_index]-lower[task_index], mc_standard_error=mc_sd[task_index],
                bootstrap_mean=bootstrap[:, task_index].mean(),
                bootstrap_bias=bootstrap[:, task_index].mean()-reference[task_index],
                zero_mean_contribution_energy=bool(np.all(mean.mean(-1) == 0) if task_index == 0 else np.all(mean[:, task_index-1] == 0)),
                bootstrap_resamples=args.bootstrap_resamples))
            halves.append(dict(**common, task=task, first_500_d=first_half[task_index],
                second_500_d=second_half[task_index], signed_difference=first_half[task_index]-second_half[task_index],
                absolute_difference=abs(first_half[task_index]-second_half[task_index]),
                full_1000_d=reference[task_index]))
        for budget in BUDGETS:
            estimates = all_d(marginals[:budget].mean(0))
            for task_index, task in enumerate(TASKS):
                discrepancy = estimates[task_index] - reference[task_index]
                convergence.append(dict(**common, task=task, permutations=budget,
                    d=estimates[task_index], finite_reference_d=reference[task_index],
                    difference_from_finite_reference=discrepancy,
                    absolute_difference_from_finite_reference=abs(discrepancy),
                    relative_difference_from_finite_reference=discrepancy/reference[task_index] if reference[task_index] != 0 else np.nan,
                    undefined_relative_difference=bool(reference[task_index] == 0)))
        for neuron in range(200):
            overall_rows = marginals[:, neuron].mean(-1)
            contributions.append(dict(**common, task='overall', neuron=neuron,
                 mean=overall_rows.mean(), monte_carlo_se=overall_rows.std(ddof=1)/np.sqrt(1000)))
            for digit in range(10):
                contributions.append(dict(**common, task=f'digit_{digit}', neuron=neuron,
                           mean=mean[neuron, digit], monte_carlo_se=se[neuron, digit]))
        quality.append(dict(**common, **metadata['validation'],
            efficiency_error=metadata['efficiency_error'], near_tie_coalition_rechecks=metadata['near_tie_rechecks'],
            zero_predicted_count_entries=metadata['score_zero_denominators'],
            intact_sample_accuracy=cc[0, 0].sum()/1024,
            intact_mean_digit_score=np.mean(cc[0, 0] / (pc[0, 0]+1e-6)),
            intact_digit_predictions=json.dumps(pc[0, 0].tolist()),
            full_lesion_sample_accuracy=cc[0, -1].sum()/1024,
            complete_vectors=1000, neurons=200, digit_tasks=10))
        seeds.append({k: record[k] for k in ['key','condition','trial','permutation_seed','validation_seed','bootstrap_seed','checkpoint_sha256','output_sha256']})
        bootstrap_models[condition].append(bootstrap)
        reference_models[condition].append(reference)
        print(f'ANALYZED {key}: {time.monotonic()-start:.1f}s', flush=True)
    frames = dict(convergence=pd.DataFrame(convergence), model_uncertainty=pd.DataFrame(uncertainty),
                  independent_halves=pd.DataFrame(halves), shapley_mean_and_se=pd.DataFrame(contributions),
                  validation=pd.DataFrame(quality), seeds=pd.DataFrame(seeds))
    group_bootstraps, summary, contrasts = {}, [], []
    for condition in bootstrap_models:
        # Independent model streams provide independent conditional MC draws.
        # Models are held fixed: this is NOT a model/population bootstrap.
        boot = np.stack(bootstrap_models[condition]).mean(0)
        estimates = np.stack(reference_models[condition])
        group_bootstraps[condition] = boot
        low, high = np.quantile(boot, [.025, .975], axis=0)
        for task_index, task in enumerate(TASKS):
            subset = frames['model_uncertainty'].query('condition == @condition and task == @task')
            split = frames['independent_halves'].query('condition == @condition and task == @task')
            summary.append(dict(condition=condition, task=task, models=10,
                mean_d=estimates[:, task_index].mean(), between_model_sd=estimates[:, task_index].std(ddof=1),
                min_d=estimates[:, task_index].min(), max_d=estimates[:, task_index].max(),
                conditional_mean_ci025=low[task_index], conditional_mean_ci975=high[task_index],
                mean_model_mc_se=subset.mc_standard_error.mean(),
                median_model_mc_interval_width=subset.interval_width.median(),
                max_model_mc_interval_width=subset.interval_width.max(),
                median_absolute_half_difference=split.absolute_difference.median(),
                max_absolute_half_difference=split.absolute_difference.max(),
                mean_bootstrap_bias=subset.bootstrap_bias.mean(),
                zero_mean_energy_models=int(subset.zero_mean_contribution_energy.sum())))
    for a, b in [('L2', 'L1'), ('None', 'L1'), ('None', 'L2')]:
        difference = group_bootstraps[a] - group_bootstraps[b]
        low, high = np.quantile(difference, [.025, .975], axis=0)
        mean_a, mean_b = np.mean(reference_models[a], axis=0), np.mean(reference_models[b], axis=0)
        for task_index, task in enumerate(TASKS):
            gap = mean_a[task_index] - mean_b[task_index]
            task_models = frames['model_uncertainty'].query('task == @task and condition in [@a, @b]')
            width = task_models.interval_width.median()
            contrasts.append(dict(comparison=f'{a} minus {b}', task=task,
                difference_of_model_means=gap, conditional_mc_ci025=low[task_index], conditional_mc_ci975=high[task_index],
                conditional_mc_se=difference[:, task_index].std(ddof=1),
                median_individual_interval_width=width,
                interval_width_over_absolute_gap=width/abs(gap) if gap != 0 else np.nan,
                undefined_width_gap_ratio=bool(gap == 0),
                conditional_interval_excludes_zero=bool(low[task_index] > 0 or high[task_index] < 0)))
    frames['condition_summary'] = pd.DataFrame(summary)
    frames['condition_contrasts'] = pd.DataFrame(contrasts)
    frames['convergence_summary'] = frames['convergence'].groupby(['condition','task','permutations'], sort=False).agg(
        models=('d','size'), mean_d=('d','mean'), between_model_sd=('d','std'),
        median_absolute_difference=('absolute_difference_from_finite_reference','median'),
        max_absolute_difference=('absolute_difference_from_finite_reference','max')).reset_index()
    for name, frame in frames.items():
        frame.to_csv(OUT / f'{name}.csv', index=False, float_format='%.12g')
    atomic_json(OUT / 'analysis_manifest.json', dict(
        bootstrap_resamples=args.bootstrap_resamples, percentile_interval=[.025,.975],
        analysis_source_sha256=sha256(Path(__file__)), statistics_source_sha256=source_hash,
        inference_manifest_sha256=sha256(OUT/'manifest.json'),
        condition_mean_interval='Monte Carlo uncertainty conditional on the ten fixed checkpoints and evaluation batch',
        between_model_variation='Sample SD of ten checkpoint D estimates; reported separately, no model resampling',
        finite_reference='1000 permutations, not exact Shapley or true D; nested estimates share permutation rows',
        fixed_budget='1000 per model; no extensions, per user instruction',
        outputs={name+'.csv': sha256(OUT/(name+'.csv')) for name in frames}))
    print('ALL STATISTICS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
