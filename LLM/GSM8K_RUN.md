Run from the repository root:

```bash
sbatch gsm8k.slurm
```

The default array has eight independent workers, with at most four running at
once. Each worker requests two A100 GPUs for 24 hours. One model is spread across
its two visible GPUs (`device_map=balanced`); this is model placement, not distributed
training. Default batch size is 8. GPU memory use and speed still need a cluster
smoke run. Reduce batch size if memory is insufficient, using a fresh output
directory when changing evaluation settings.

For just the random control, with four workers and at most two at a time:

```bash
sbatch --array=0-3%2 gsm8k.slurm --conditions random
```

For a short first run:

```bash
sbatch --array=0-0 --time=01:00:00 gsm8k.slurm \
  --conditions random --counts 0 5 --num-trials 1 --num-examples 32 \
  --max-runtime-hours 0.8 --output-dir LLM/results/gsm8k_smoke
```

Each condition/removal-count/trial has its own checkpoint and lock. Progress is
saved after every batch. Resubmit the identical command to continue after the
24-hour limit; only an interrupted batch may need repeating. The runner stops
starting batches after 23.8 hours, or after Slurm's advance signal. A single
batch may run beyond that soft deadline. Checkpoints are replaced atomically.
Locks prevent overlapping submissions from writing the same evaluation. Array
indices must start at zero and be contiguous. You can change the number of
workers when resubmitting. All workers in one experiment must share the same
conditions, counts, and trial count so their work assignments partition correctly.

Fresh results go to `LLM/results/gsm8k_seeded`, preserving the historical JSONs.
The output directory records input hashes, software versions, GPU count and
settings, and rejects incompatible workers. Keep software/model files unchanged
when resuming. Each trial has a seeded three-shot prompt, identical across high,
low and random conditions. Trial prompts vary across trials. Random removal uses
one seeded uniform ordering of all experts per trial, with nested subsets for
each removal count. Generation is explicitly greedy; seeds do not guarantee
bitwise identity across different hardware or CUDA implementations. Existing
historical results used different settings and should be compared separately.
CSV rows are aligned by metric, layer and expert before combining contributions
with the original 4:3 weights; both `_accuracy` and plain metric names work.

After workers stop, export completed trials to the old plotting format:

```bash
uv run --frozen --group llm python LLM/script.py --merge-only
```

Use the same `--output-dir`, `--conditions`, `--counts`, and `--num-trials` if you
changed them. Export prints completeness, excludes unfinished evaluations, and
writes `accuracies_{low,high,random}.json`, `time_taken_*.json`, and matching
`trial_ids_*.json` inside the output directory. Checkpoints remain the resume
source. No model or GPU is needed for export. The old notebook still reads the
historical files in `LLM/`; point it to the exported files to plot the new runs.
