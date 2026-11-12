"""Resumable, independently sharded GSM8K expert-removal evaluations."""
import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
from itertools import product
import json
from pathlib import Path
import signal
import tempfile
import time

SCRIPT_DIR = Path(__file__).resolve().parent
lesion_dictionary = defaultdict(list)
STOP = False


def atomic_json(path, value):
    # A distinct file per write also avoids collisions across cluster nodes.
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp",
                                     delete=False) as f:
        temporary = Path(f.name)
        try:
            f.write(json.dumps(value, indent=2) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_for(seed, trial, batch=0):
    # Stable across processes, conditions, removal counts and resubmissions.
    import numpy as np
    return int(np.random.SeedSequence([seed, trial, batch]).generate_state(1)[0])


def removed_experts(population, count, condition, seed, trial):
    if condition == "random":
        import random
        order = random.Random(seed_for(seed, trial, 1000000)).sample(population, len(population))
    else:
        order = population if condition == "low" else population[::-1]
    return tuple(order[:count])


def rank_experts(paths):
    # Align by metric/layer/expert, rather than relying on CSV row ordering.
    tables = []
    for path in paths:
        table = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                metric = row["Performance Metric"].removesuffix("_accuracy")
                if metric not in ("addition", "subtraction", "multiplication"):
                    continue
                key = (metric, int(row["Layer"]), int(row["Expert"]))
                if key in table:
                    raise ValueError(f"Duplicate entry in {path}: {key}")
                table[key] = float(row["Contribution"])
        tables.append(table)
    if not tables[0] or tables[0].keys() != tables[1].keys():
        raise ValueError("Ranking CSVs must contain matching arithmetic entries")
    values = defaultdict(list)
    for (metric, layer, expert), first in tables[0].items():
        values[layer, expert].append((4 * first + 3 * tables[1][metric, layer, expert]) / 7)
    if any(len(v) != 3 for v in values.values()):
        raise ValueError("Each expert needs all three arithmetic metrics")
    return tuple(sorted(values, key=lambda k: (sum(values[k]) / 3, k)))


def model_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralConfig, MixtralDecoderLayer, MixtralForCausalLM,
        MixtralModel, MixtralSparseMoeBlock,
    )
    class LesionedMixtralSparseMoeBlock(MixtralSparseMoeBlock):
        def __init__(self, config, layer_idx):
            super().__init__(config)
            self.layer_idx = layer_idx

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            ## Most of the code is copied from the original implementation
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if self.training and self.jitter_noise > 0:
                hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
            hidden_states = hidden_states.view(-1, hidden_dim)
            # router_logits: (batch * sequence_length, n_experts)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            routing_weights = routing_weights.to(hidden_states.dtype)

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
            )

            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

            # Loop over all available experts in the model and perform the computation on each expert
            for expert_idx in range(self.num_experts):
                # if the current expert is lesioned, skip it. This is the only change I made
                if expert_idx in lesion_dictionary[self.layer_idx]:
                    continue

                expert_layer = self.experts[expert_idx]
                idx, top_x = torch.where(expert_mask[expert_idx])

                # Index the correct hidden states and compute the expert hidden state for
                # the current expert. We need to make sure to multiply the output hidden
                # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
                current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

                # However `index_add_` only support torch tensors for indexing so we'll use
                # the `top_x` tensor here.
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

    class LesionedMixtralDecoderLayer(MixtralDecoderLayer):
        def __init__(self, config: MixtralConfig, layer_idx: int):
            super().__init__(config, layer_idx)
            self.block_sparse_moe = LesionedMixtralSparseMoeBlock(config, layer_idx)

    class LesionedMixtralModel(MixtralModel):
        def __init__(self, config: MixtralConfig):
            super().__init__(config)
            self._no_split_modules = ["LesionedMixtralDecoderLayer"]
            self.layers = nn.ModuleList(
                [LesionedMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.post_init()

    class LesionedMixtralForCausalLM(MixtralForCausalLM):
        _tied_weights_keys = ["lm_head.weight"]

        def __init__(self, config):
            super().__init__(config)
            self.model = LesionedMixtralModel(config)
            # Initialize weights and apply final processing
            self.post_init()


    return LesionedMixtralForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results/gsm8k_seeded")
    parser.add_argument("--conditions", nargs="+", choices=["low", "high", "random"], default=["low", "high", "random"])
    parser.add_argument("--counts", type=int, nargs="+", default=[0, 1, *range(5, 75, 5)])
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2810)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shots", type=int, default=3)
    parser.add_argument("--num-examples", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--model", default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--max-runtime-hours", type=float, default=23.8)
    parser.add_argument("--ranking-csvs", nargs=2, type=Path, default=[SCRIPT_DIR / "shapley_analysis_20240614-002232.csv", SCRIPT_DIR / "shapley_analysis_20240805-062942.csv"])
    parser.add_argument("--merge-only", action="store_true", help="Export completed trials into plotting JSON files; run after workers stop")
    args = parser.parse_args()
    if args.num_trials <= 0 or args.batch_size <= 0 or args.num_shots < 0:
        parser.error("Trials and batch size must be positive; shots must be nonnegative")
    if args.num_examples == 0 or args.num_examples < -1 or args.max_new_tokens <= 0:
        parser.error("Examples must be -1 or positive; token limit must be positive")
    if not 0 <= args.shard_id < args.num_shards:
        parser.error("Require 0 <= shard-id < num-shards")
    if any(n < 0 for n in args.counts) or args.max_runtime_hours <= 0:
        parser.error("Counts must be nonnegative and runtime must be positive")
    args.counts = sorted(set(args.counts))
    args.conditions = sorted(set(args.conditions))
    return args


def task_path(output_dir, condition, count, trial):
    return output_dir / "checkpoints" / f"{condition}_n{count:03d}_trial{trial:03d}.json"


def merge_results(args):
    for condition in args.conditions:
        accuracies, timings = {}, {}
        identities = {}
        for count in args.counts:
            complete = []
            for trial in range(args.num_trials):
                path = task_path(args.output_dir, condition, count, trial)
                if path.exists():
                    state = json.loads(path.read_text())
                    if state["complete"]:
                        complete.append((trial, state))
            if complete:
                key = str(count)
                accuracies[key] = [s["correct"] / s["total"] for _, s in complete]
                timings[key] = [s["seconds"] for _, s in complete]
                identities[key] = [t for t, _ in complete]
            print(f"{condition} removed={count}: {len(complete)}/{args.num_trials} complete", flush=True)
        atomic_json(args.output_dir / f"accuracies_{condition}.json", accuracies)
        atomic_json(args.output_dir / f"time_taken_{condition}.json", timings)
        atomic_json(args.output_dir / f"trial_ids_{condition}.json", identities)


def main():
    args = parse_args()
    if args.merge_only:
        merge_results(args)
        return
    started = time.monotonic()
    def request_stop(signum, frame):
        global STOP
        STOP = True
        print("Stop requested; saving after the current batch", flush=True)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    ranking = rank_experts(args.ranking_csvs)
    # Check configuration under a lock: never silently mix different experiments.
    config = {k: v for k, v in vars(args).items() if k not in {
        "output_dir", "conditions", "counts", "num_trials", "num_shards", "shard_id",
        "max_runtime_hours", "merge_only", "ranking_csvs"}}
    config["protocol"] = "v2_greedy_seeded_prompt_per_trial_left_padding"
    inputs = [*args.ranking_csvs, SCRIPT_DIR / "GSM8K/data/train.jsonl", SCRIPT_DIR / "GSM8K/data/test.jsonl"]
    config["input_sha256"] = [hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs]
    import torch
    import transformers
    config["torch_version"] = torch.__version__
    config["transformers_version"] = transformers.__version__
    config["gpu_count"] = torch.cuda.device_count()
    with (args.output_dir / "config.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        config_path = args.output_dir / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("Output directory has different settings; use a new --output-dir")
        if not config_path.exists():
            atomic_json(config_path, config)
    tasks = list(product(args.conditions, args.counts, range(args.num_trials)))
    tasks = tasks[args.shard_id::args.num_shards]
    tasks = [(c, n, t) for c, n, t in tasks if not (
        task_path(args.output_dir, c, n, t).exists() and
        json.loads(task_path(args.output_dir, c, n, t).read_text())["complete"])]
    if not tasks:
        print("All assigned evaluations are complete", flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The quantized Mixtral evaluation requires CUDA")
    from transformers import AutoTokenizer, BitsAndBytesConfig, set_seed
    from GSM8K import dataset
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = model_class().from_pretrained(
        args.model, device_map=args.device_map, torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    ).eval()
    print(f"Model placement: {model.hf_device_map}", flush=True)
    population = tuple(product(range(model.config.num_hidden_layers), range(model.config.num_local_experts)))
    if set(ranking) != set(population) or max(args.counts) > len(population):
        raise ValueError("Expert ranking/counts do not match the loaded model")
    input_device = model.get_input_embeddings().weight.device
    for condition, count, trial in tasks:
        if STOP or time.monotonic() - started >= args.max_runtime_hours * 3600:
            break
        path = task_path(args.output_dir, condition, count, trial)
        with path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue  # Another independently submitted worker owns this task.
            selected = removed_experts(population if condition == "random" else ranking,
                                       count, condition, args.seed, trial)
            state = json.loads(path.read_text()) if path.exists() else {
                "condition": condition, "count": count, "trial": trial,
                "removed_experts": selected, "correct": 0, "total": 0,
                "seconds": 0.0, "complete": False,
            }
            if state["complete"]:
                continue
            # Each trial uses a different reproducible prompt, shared across conditions.
            data = dataset.GSM8KDataset(tokenizer, num_shots=args.num_shots,
                num_examples=args.num_examples, seed=seed_for(args.seed, trial))
            lesion_dictionary.clear()
            for layer, expert in selected:
                lesion_dictionary[layer].append(expert)
            with torch.inference_mode():
                for offset in range(state["total"], len(data), args.batch_size):
                    if STOP or time.monotonic() - started >= args.max_runtime_hours * 3600:
                        break
                    batch_start = time.monotonic()
                    set_seed(seed_for(args.seed, trial, offset // args.batch_size))
                    batch = [data[i] for i in range(offset, min(offset + args.batch_size, len(data)))]
                    prompts, answers = zip(*batch)
                    encoded = tokenizer(list(prompts), return_tensors="pt", padding=True).to(input_device)
                    generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.eos_token_id)
                    completions = tokenizer.batch_decode(generated[:, encoded.input_ids.shape[1]:], skip_special_tokens=True)
                    predictions = [dataset.extract_answer(text) for text in completions]
                    state["correct"] += sum(p == y for p, y in zip(predictions, answers))
                    state["total"] += len(batch)
                    state["seconds"] += time.monotonic() - batch_start
                    state["complete"] = state["total"] == len(data)
                    atomic_json(path, state)
                    print(f"{condition} n={count} trial={trial}: {state['total']}/{len(data)}; accuracy={state['correct']/state['total']:.4f}", flush=True)
    print("Worker finished. Resubmit with the same settings to continue.", flush=True)


if __name__ == "__main__":
    main()
