# MLP sampling stability of D

This is a **200-neuron stability check**, using the existing ten checkpoints per
condition (L1, L2, unregularized), without retraining. The user-approved sampling
budget is exactly 1,000 independent permutations per model; no extensions are run.

Read [the illustrated PDF](results/ILLUSTRATED_REPORT.pdf) or its
[Markdown version](results/ILLUSTRATED_REPORT.md). All four figures are included.
The publication archive contains PDF, editable-text SVG, and 600-dpi PNG exports.
Styling follows `MLP/visual_config.py`.
`results/NUMERICAL_SUMMARIES.zip` bundles all nine CSV tables, manifests, the
fixed evaluation batch and documentation. Complete per-permutation arrays and
bootstrap distributions remain alongside the summaries as separate NPZ files.

## Fixed evaluation protocol

One seeded, shuffled 1,024-image MNIST test batch is saved and shared across all
models, coalitions and permutation budgets. The batch has all ten digits. This
holds the value function fixed, as required for isolating permutation uncertainty.
The historical runner used a shuffled loader for each objective invocation, so
this experiment is not a reproduction of those historical stochastic batches.

**The implemented original digit score is precision.**
`evaluation.evaluate_model` calls `score_fn(predictions, labels)` while
`accuracy_each_class` names its arguments `(targets, preds)`. Consequently the
original confusion-matrix row totals count predicted digits, and its score is
`TP_c / (predicted_count_c + 1e-6)`. We preserve that argument order, including
zero for unpredicted digits. We do not replace it with recall or pooled accuracy.
Overall D uses the mean signed contribution over these ten digit scores before
squaring and calculating entropy; it does not average ten D values.

The intervention zeros outputs of the first linear layer before LeakyReLU. The
complete set of 200 hidden neurons remains in the game, and the output bias
remains active even when all hidden neurons are removed. Models and activations
use float32, with CUDA TF32 matrix multiplication disabled. Scores, contributions
and statistical calculations use float64.

## Reproduce

From the repository root, using the existing environment:

```sh
/home/shrey/miniconda3/envs/msa-xai/bin/python MLP/run_sampling_stability.py --device cuda:2
/home/shrey/miniconda3/envs/msa-xai/bin/python MLP/analyze_sampling_stability.py
/home/shrey/miniconda3/envs/msa-xai/bin/python MLP/plot_sampling_stability.py
/home/shrey/miniconda3/envs/msa-xai/bin/python MLP/sampling_stability/build_report.py
/home/shrey/miniconda3/envs/msa-xai/bin/python -m pytest MLP/sampling_stability/test_statistics.py -q
```

The experiment was launched in tmux as `mlp_sampling_stability`, with progress in
`cache/run.log`; that session exits on completion. To launch a fresh/resumed run:

```sh
tmux new-session -d -s mlp_sampling_stability '/home/shrey/miniconda3/envs/msa-xai/bin/python -u /home/shrey/MSA-XAI/MLP/run_sampling_stability.py --device cuda:2 > /home/shrey/MSA-XAI/MLP/sampling_stability/cache/run.log 2>&1'
```

Inference checkpoints every 100 permutations and skips completed model files.
Source, input and configuration identity checks reject incompatible resumptions.
Statistical bootstrap arrays are cached with source/input hashes. GPU access is
needed only for inference; analysis and figure/report regeneration use the CPU.
Use the listed environment; `requirements.txt` records its relevant versions.

## Sampling and uncertainty

For each model a dedicated NumPy PCG64 stream samples 1,000 independent neuron
removal orders. All marginal vectors, not just their averages, are retained.
The implementation explicitly evaluates ordered removal paths so stored rows
retain reproducible order and identity. This serves the role of msapy's
`save_permutations=True` while enabling batched evaluation and resumability;
the installed library does not have a `return_shapley_table` argument.

Budget estimates use the first 100, 250, 500, 750 and 1,000 rows. The final estimate
is a **finite-sample reference**, not an exact Shapley/D value. Nested errors are
dependent and are zero at 1,000 by construction. The two 500-row halves contain
independent sampled permutations.

Each of 10,000 bootstrap draws resamples whole permutation rows, using shared
multinomial weights across all neurons and digit tasks. Every draw recomputes the
mean signed Shapley contributions and then D, including the overall task average.
Intervals are the 2.5th and 97.5th percentiles; bootstrap SD and bias are also
saved. Percentile intervals can be displaced from the point by nonlinear
plug-in bias. No neuron, digit, image or model is resampled as a permutation
replicate. Contribution SEs also use permutation rows and retain task covariance
when computing the overall contribution.

Condition-mean intervals average independent within-model bootstrap draws across
the ten fixed models. Differences between those means have **conditional Monte
Carlo intervals**. They do not include uncertainty over training runs or evaluation
images and are not population-level significance tests. Between-model sample SD
uses `ddof=1` and is reported separately. Ratios with zero denominators are marked
undefined. Zero-energy mean contributions follow the original rule D=0.

## Files

| File | Contents |
|---|---|
| `results/evaluation_batch.npz` | Exact image indices, normalized images and labels |
| `results/permutations_*.npz` | One file per model: `permutations` (1000, 200), `marginals` (1000, 200, 10), `correct_counts` and `predicted_counts` (1000, 201, 10), JSON `metadata` |
| `results/bootstrap_*.npz` | Per-model D resamples (10000, 11), ordered overall then digits 0-9; identity metadata |
| `results/convergence.csv` | Each model/task at every requested budget and differences from finite reference |
| `results/convergence_summary.csv` | Descriptive condition summaries of budget estimates |
| `results/model_uncertainty.csv` | Per-model/task D, 95% MC interval, width, MC SE, bootstrap bias, zero-energy flag |
| `results/independent_halves.csv` | First/second 500-row D and their differences |
| `results/condition_summary.csv` | Means, between-model SD/range, conditional-mean intervals, MC-width and half-run summaries |
| `results/condition_contrasts.csv` | Differences between condition means with conditional MC intervals and width/gap ratios |
| `results/shapley_mean_and_se.csv` | All neuron/task mean contributions and permutation MC SEs |
| `results/seeds.csv` | Per-model sampling, validation and bootstrap seeds and artifact hashes |
| `results/validation.csv` | Original-scorer comparisons, efficiency residuals, zero score denominators and endpoints |
| `results/*manifest.json` | Source/input hashes, device/precision, sampling protocol and figure provenance |

When reading CSVs with pandas, pass `keep_default_na=False` to preserve the
literal unregularized condition label `None`; otherwise pandas interprets that
label as missing data. Undefined numeric ratios have explicit Boolean flags.

Permutation column entries give the neuron removed at that position. Marginal
axis 1 is in **neuron-ID order**, not removal order. Count axis 1 has intact at
index 0, then one more neuron removed at each subsequent index. Predictions can
change class frequencies, so predicted-count denominators must be retained.

The accelerated evaluator masks precomputed hidden activations and uses the same
final affine layer. Because LeakyReLU(0)=0, this preserves the zero intervention.
Near-tie coalitions (top-two logit margin below 1e-4) are rechecked with the
original model forward to protect discrete predictions against batched-GEMM
rounding. Each model is additionally checked on 128 independent validation
coalitions against the original forward and sklearn-based score. Every
permutation is checked for telescoping efficiency and consistent endpoints.
