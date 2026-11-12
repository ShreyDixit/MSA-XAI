# Larger DCGAN quantitative analysis

The main entry point is `DCGAN/quantify.py`. It reuses only the historical larger-network
Shapley Modes, checks their compatibility, and computes spatial intervention agreement
(Option 2) and signed-map spatial scale (Option 3). See `results/REPORT.md` for the
generated methods/results and `results/manifest.json` for exact input identities.

The illustrated report is `results/ILLUSTRATED_REPORT.pdf`. Pages 1-4 explain the
results in plain language with all six figures; pages 5-6 retain the equations,
controls, full uncertainty tables, numerical checks, and reproducibility details.
Matching Markdown and editable LaTeX are alongside it. Redraw the publication
figures and rebuild the report without rerunning inference:

```sh
python DCGAN/plot_quantification.py
python DCGAN/quantification/build_illustrated_report.py
```

PDF generation uses an installed `pdflatex` with geometry, graphicx, booktabs,
hyperref, amsmath, and Latin Modern fonts. The builder uses Python's standard library and
checks that every generated figure is included and no content overflows.
`results/PUBLICATION_FIGURES.zip` bundles all six figures in PDF, SVG, and PNG
formats together with their styling and qualitative-selection provenance.

`plot_quantification.py` imports the shared palette and visual settings from
`MLP/visual_config.py` (identical to `LLM/visual_config.py` when inspected). It uses
8-point labels, thin axes, centimetre-based sizing, embedded TrueType PDF fonts,
editable SVG text, and 600-dpi PNG exports. It uses Atkinson Hyperlegible when
available, otherwise the DejaVu Sans font embedded in the reference experiment
PDFs. `figure_manifest.json` records the actual font, style source hash, and input
CSV hashes. Figure layers 1-5 correspond to saved CSV/model layers 0-4.

The qualitative plate uses image 30 and filters 72/50/7/12/8, from the pre-existing
notebook's candidate examples. The choices illustrate visible spatial structure
and were not selected by agreement score; they are not a representative sample.
`qualitative_examples.json` records IDs and raw heatmap maxima. Each display
heatmap is divided by its own maximum; raw maps and all statistical calculations
remain unnormalized. The purple overlay shows the exact selected top-20% pixels.

Run from the repository root using the project's Python environment:

```sh
python DCGAN/quantify_gpu_controls.py --device cuda:0
python DCGAN/quantify_sensitivity.py
python DCGAN/plot_quantification.py
python DCGAN/quantification/build_illustrated_report.py
```

The environment used here is `/home/shrey/miniconda3/envs/msa-xai/bin/python`.
Dependencies are already in the project's Poetry environment: NumPy, pandas, SciPy,
matplotlib, PyTorch, torchvision, msapy, imageio, fastprogress, and threadpoolctl.
The exact installed versions are recorded in the result manifest. Tests additionally
need pytest. No training images or network downloads are required.
The installed dependency versions are also pinned in `requirements.txt`.

`quantify_gpu_controls.py` runs the same main pipeline with an accelerated float64
CUDA permutation gather/reduction. It checks agreement with the CPU definition on
real-sized maps before execution and writes `spatial_control_backend.json` with
the backend's source hash and observed discrepancy. It uses the identical seeded
permutation banks. The CPU reference entry point is `quantify.py` (slower for the
shuffle control); all CLI arguments and cache identities are shared.

Defaults: all 32 saved latents; all 1,984 hidden filters; 100 seeded spatial shuffles;
all other same-layer filters as mismatch references; 10,000 image-level bootstrap
resamples; seed 2810; inference batch 32, float32 and historical cuDNN/TF32 defaults.
The default device is `cuda:0`. CUDA must be accessible to the process. The original
checkpoint and modes are never modified. No new Shapley estimates are generated.

Stages can be run separately, with identical settings and input artifacts:

```sh
python DCGAN/quantify.py --stage validate
python DCGAN/quantify.py --stage ablations
python DCGAN/quantify.py --stage modes
python DCGAN/quantify.py --stage agreement
python DCGAN/quantify.py --stage report
```

`ablations` repeats validation and resumes completed filters. `agreement` resumes
completed layers; incomplete layers are recomputed. `modes` recomputes masks and
spectral metrics. `report` redraws figures and rebuilds summaries without using CUDA.
After the core analysis/report or sensitivity stages, rerun `plot_quantification.py`
to apply the final publication style, followed by the illustrated report builder.
Input and code SHA-256 values and run settings guard against incompatible cache
reuse. Use fresh `--cache` and `--output` directories when changing settings or code.
For a CPU validation, use fresh directories with `--device cpu`; numerical behavior
can differ from the historical CUDA implementation and strict compatibility may fail.

The approximately 24 GB pickle is read with read-only memory mapping using the
protocol-5 NumPy payload; it is not copied into a second 24 GB array. Accesses remain
per filter. Only load trusted project pickles. The raw ablation cache is approximately
12.5 GB; packed masks occupy approximately 130 MB. Cache files are ignored by Git.
CSV tables, figures, methods/results, latent values, and provenance remain reviewable.

Raw outputs in `cache/ablations_layer_L.npy` have shape `(filter, image, RGB, y, x)`;
completion flags are in `cache/ablations_layer_L_done.npy`. These are the reusable
single-filter intervention outputs. `baseline.npy` is the unlesioned image batch;
`latents.npy` preserves the exact latent values and dtype. All image indices are
zero-based and follow `fixed_noise.t`; units follow `results/unit_mapping.csv`.

The spatial score uses exactly 3,277/16,384 pixels, randomizing boundary ties with
recorded seeds. Spatially shuffling a map is equivalent to uniformly permuting its
top-pixel mask. Its exact expected capture is 3,277/16,384; empirical mean/SD/quantiles
from 100 permutations are also saved. Mismatched capture is the exact average over
every other eligible filter in the same image/layer, using summed masks to avoid
quadratic enumeration. All-zero attribution maps are excluded as reference masks.
Zero image change and zero attribution make the target score undefined. Near-zero
positive change is flagged rather than silently dropped.

Spectral centroids use signed maps, per-channel spatial demeaning, channel-summed
power, a full FFT, and radial frequencies normalized by the 2-D Nyquist corner
`sqrt(0.5**2 + 0.5**2)` cycles/pixel. Non-DC zero energy gives NaN. The primary result
is untapered; a Hann-taper sensitivity result is included. Final output-channel
players are stored and plotted separately from the hidden-layer trend.

The bootstrap sampling unit is the image, after averaging filters within each
image/layer. Gains are paired differences. Intervals do not quantify historical
Shapley Monte Carlo error, variation across checkpoints, or training seeds.

`DCGAN/quantify_sensitivity.py` checks spatial results after excluding targets whose
mean absolute image change is below 10⁻⁶ or 10⁻⁵. This addresses sub-micro-unit
cuDNN repeat differences without redefining the primary zero-change rule. It needs
no GPU and writes `results/SENSITIVITY.md`, image/layer CSVs, figures, and a manifest
with the source CSV and script hashes. See this supplementary report alongside
`results/REPORT.md` when assessing agreement for very small interventions.

Metric tests:

```sh
PYTHONPATH=DCGAN MPLCONFIGDIR=/tmp/dcgan-quantification-matplotlib \
    python -m pytest -q DCGAN/quantification/test_metrics.py
```
