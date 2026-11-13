#!/usr/bin/env python3
"""Resumable fixed-batch, zero-lesion permutation experiment. No retraining."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torchvision.datasets import MNIST
from sampling_stability.statistics import marginals_from_counts, score_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'sampling_stability'
CONDITIONS = [('L1', 'l1 regularization'), ('L2', 'l2 regularization'), ('None', 'none regularization')]
MASTER_SEED = 2810
PERMUTATIONS = 1000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def atomic_npz(path, **arrays):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix('.tmp').replace(path)


def derived_seed(condition, trial, stream):
    return int(np.random.SeedSequence([MASTER_SEED, condition, trial, stream]).generate_state(1)[0])


def model_load(path, device):
    model = torch.load(path, map_location=device, weights_only=False).eval()
    assert model.layer1.out_features == 200
    assert model.layer1.weight.dtype == torch.float32
    assert isinstance(model.layer2[0], torch.nn.LeakyReLU)
    assert float(model.layer2[0].negative_slope) == 0.01
    return model


def counts_from_predictions(predictions, labels):
    membership = predictions[..., None] == torch.arange(10, device=labels.device)
    correct = predictions == labels
    predicted_counts = membership.sum(dim=-2)
    correct_counts = (membership & correct[..., None]).sum(dim=-2)
    return correct_counts, predicted_counts


@torch.no_grad()
def coalition_counts(model, inputs, hidden, labels, masks):
    # Masking after LeakyReLU is algebraically identical to masking its input:
    # f(0)=0 and this activation is elementwise. Keep the learned output bias.
    outputs = torch.nn.functional.linear(hidden[None] * masks[:, None],
                                          model.layer2[1].weight, model.layer2[1].bias)
    predictions = outputs.argmax(dim=-1)
    # Batched GEMMs may round differently. Re-evaluate near-tie coalitions using
    # the original forward path, so accuracy thresholds retain canonical behavior.
    top_two = outputs.topk(2, dim=-1).values
    suspect = ((top_two[..., 0] - top_two[..., 1]) < 1e-4).any(dim=-1)
    for index in torch.where(suspect)[0].tolist():
        lesions = torch.where(masks[index] == 0)[0].tolist()
        predictions[index] = model(inputs, lesions, lesion_mode='zero').argmax(-1)
    correct, predicted = counts_from_predictions(predictions, labels)
    return correct.cpu().numpy().astype(np.int16), predicted.cpu().numpy().astype(np.int16), int(suspect.sum())


@torch.no_grad()
def validate(model, inputs, labels, hidden, seed):
    from evaluation import accuracy_each_class, evaluate_model
    from torch.utils.data import DataLoader, TensorDataset
    rng = np.random.default_rng(seed)
    masks = np.ones((128, 200), dtype=np.float32)
    for row in masks:
        row[rng.choice(200, size=int(rng.integers(0, 201)), replace=False)] = 0
    masks[0] = 1
    masks[1] = 0
    batched_correct, batched_predicted, suspect = coalition_counts(model, inputs, hidden, labels,
                                               torch.as_tensor(masks, device=inputs.device))
    max_score_error = 0.
    mismatched_count_entries = 0
    for index, mask in enumerate(masks):
        lesions = np.flatnonzero(mask == 0).tolist()
        predictions = model(inputs, lesions, lesion_mode='zero').argmax(-1)
        correct, predicted = counts_from_predictions(predictions, labels)
        mismatched_count_entries += int(np.count_nonzero(correct.cpu().numpy() != batched_correct[index]))
        mismatched_count_entries += int(np.count_nonzero(predicted.cpu().numpy() != batched_predicted[index]))
        # Call the original sklearn-based scorer with its original argument order.
        original = accuracy_each_class(predictions.cpu(), labels.cpu())
        candidate = score_counts(batched_correct[index], batched_predicted[index])
        max_score_error = max(max_score_error, float(np.max(np.abs(original - candidate))))
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=len(labels), shuffle=False)
    original_endpoint = evaluate_model(model, loader, [], 1, accuracy_each_class)
    assert np.array_equal(original_endpoint, score_counts(batched_correct[0], batched_predicted[0]))
    assert mismatched_count_entries == 0, (mismatched_count_entries, max_score_error)
    assert max_score_error == 0, max_score_error
    return dict(coalitions=128, mismatched_count_entries=mismatched_count_entries,
                maximum_score_error=max_score_error, near_tie_rechecks=suspect)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cuda:2')
    parser.add_argument('--coalition-chunk', type=int, default=128)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(MASTER_SEED)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This run requires the requested CUDA device.')
    cache = OUT / 'cache'
    results = OUT / 'results'
    cache.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    dataset = MNIST(str(ROOT / 'data'), train=False, download=False)
    # Same operation as a shuffled DataLoader first batch; dedicated saved seed.
    batch_seed = derived_seed(0, 0, 0)
    generator = torch.Generator().manual_seed(batch_seed)
    indices = torch.randperm(len(dataset), generator=generator)[:1024].numpy()
    inputs = dataset.data[indices].flatten(1).to(device) / 255
    labels = dataset.targets[indices].to(device)
    batch_path = results / 'evaluation_batch.npz'
    if batch_path.exists():
        with np.load(batch_path) as old:
            assert np.array_equal(old['indices'], indices)
    else:
        atomic_npz(batch_path, indices=indices, images=inputs.cpu().numpy(), labels=labels.cpu().numpy())
    source_files = [Path(__file__), OUT / 'statistics.py', ROOT / 'model.py', ROOT / 'evaluation.py', ROOT / 'data.py']
    identity = dict(master_seed=MASTER_SEED, batch_seed=batch_seed,
                    batch_sha256=sha256(batch_path), permutations=PERMUTATIONS,
                    source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_files},
                    device=args.device, float_precision='float32 model / float64 scores and statistics',
                    cuda_matmul_allow_tf32=False, coalition_chunk=args.coalition_chunk)
    manifest_path = results / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['identity'] != identity:
            raise RuntimeError('Run identity changed; retain old results in a separate directory before rerunning.')
    else:
        manifest = dict(identity=identity, created_utc=datetime.now(timezone.utc).isoformat(),
            python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
            cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
            original_value_function='diagonal(confusion_matrix(predictions, labels)) / (row_sum + 1e-6)',
            overall='D of arithmetic mean of signed contributions across the ten digit scores',
            intervention='Zero layer1 outputs before LeakyReLU; all 200 hidden neurons are players',
            batch_policy='One seeded shuffled 1024-image MNIST test batch shared by all models and coalitions',
            digit_counts=np.bincount(labels.cpu().numpy(), minlength=10).tolist(),
            dataset_sha256={p.name: sha256(p) for p in (ROOT / 'data/MNIST/raw').glob('t10k-*-ubyte')},
            permutation_policy='Independent numpy PCG64 streams per model; removal permutations sampled iid, no antithetic pairing',
            models=[])
    for condition_index, (condition, folder) in enumerate(CONDITIONS):
        for trial in range(10):
            key = f'{condition}_{trial:02d}'
            model_path = ROOT / 'models' / folder / '200' / f'trial_{trial}.pt'
            checkpoint_hash = sha256(model_path)
            permutation_seed = derived_seed(condition_index, trial, 1)
            validation_seed = derived_seed(condition_index, trial, 2)
            bootstrap_seed = derived_seed(condition_index, trial, 3)
            seed_record = dict(key=key, condition=condition, trial=trial,
                    checkpoint=str(model_path.relative_to(ROOT)), checkpoint_sha256=checkpoint_hash,
                    permutation_seed=permutation_seed, validation_seed=validation_seed, bootstrap_seed=bootstrap_seed)
            old_record = next((m for m in manifest['models'] if m['key'] == key), None)
            if old_record:
                assert all(old_record[k] == v for k, v in seed_record.items())
            else:
                manifest['models'].append(seed_record)
            if not args.pilot:
                atomic_json(manifest_path, manifest)
            output_path = results / f'permutations_{key}.npz'
            if output_path.exists():
                print(f'SKIP {key}: complete saved vectors', flush=True)
                continue
            rng = np.random.default_rng(permutation_seed)
            permutations = np.stack([rng.permutation(200) for _ in range(PERMUTATIONS)]).astype(np.int16)
            model = model_load(model_path, device)
            hidden = model.layer2[0](model.layer1(inputs)).detach()
            validation = validate(model, inputs, labels, hidden, validation_seed)
            progress_path = cache / f'{key}_progress.npz'
            correct_counts = np.zeros((PERMUTATIONS, 201, 10), dtype=np.int16)
            predicted_counts = np.zeros_like(correct_counts)
            completed = 0
            rechecks = 0
            if progress_path.exists() and not args.pilot:
                with np.load(progress_path) as progress:
                    assert str(progress['identity']) == json.dumps(identity, sort_keys=True)
                    assert np.array_equal(progress['permutations'], permutations)
                    correct_counts[:] = progress['correct_counts']
                    predicted_counts[:] = progress['predicted_counts']
                    completed = int(progress['completed'])
                    rechecks = int(progress['near_tie_rechecks'])
            start = time.monotonic()
            limit = 8 if args.pilot else PERMUTATIONS
            for permutation_index in range(completed, limit):
                permutation = permutations[permutation_index]
                masks = np.ones((201, 200), dtype=np.float32)
                masks[np.arange(1, 201), permutation] = 0
                np.minimum.accumulate(masks, axis=0, out=masks)
                for begin in range(0, 201, args.coalition_chunk):
                    end = min(begin + args.coalition_chunk, 201)
                    tensor_masks = torch.as_tensor(masks[begin:end], device=device)
                    cc, pc, n_rechecks = coalition_counts(model, inputs, hidden, labels, tensor_masks)
                    correct_counts[permutation_index, begin:end] = cc
                    predicted_counts[permutation_index, begin:end] = pc
                    rechecks += n_rechecks
                if not args.pilot and ((permutation_index + 1) % 100 == 0 or permutation_index == PERMUTATIONS - 1):
                    atomic_npz(progress_path, permutations=permutations, correct_counts=correct_counts,
                               predicted_counts=predicted_counts, completed=permutation_index + 1,
                               near_tie_rechecks=rechecks, identity=json.dumps(identity, sort_keys=True))
                    print(f'{key} {permutation_index+1}/1000; elapsed={time.monotonic()-start:.1f}s; rechecks={rechecks}', flush=True)
            if args.pilot:
                print(json.dumps(dict(pilot_model=key, seconds=time.monotonic()-start,
                                 permutations=limit, validation=validation, near_tie_rechecks=rechecks,
                                 digit_counts=manifest['digit_counts'])), flush=True)
                return
            marginals = marginals_from_counts(correct_counts, predicted_counts, permutations)
            scores = score_counts(correct_counts, predicted_counts)
            efficiency_error = float(np.max(np.abs(marginals.sum(1) - (scores[:, 0] - scores[:, -1]))))
            assert efficiency_error < 1e-12
            assert np.all(correct_counts[:, 0] == correct_counts[0, 0])
            assert np.all(predicted_counts[:, 0] == predicted_counts[0, 0])
            assert np.all(predicted_counts.sum(-1) == 1024)
            metadata = dict(**seed_record, identity=identity, validation=validation,
                            near_tie_rechecks=rechecks, efficiency_error=efficiency_error,
                            score_zero_denominators=int((predicted_counts == 0).sum()),
                            elapsed_seconds=time.monotonic()-start)
            atomic_npz(output_path, permutations=permutations, marginals=marginals,
                       correct_counts=correct_counts, predicted_counts=predicted_counts,
                       metadata=json.dumps(metadata, sort_keys=True))
            old_record = next(m for m in manifest['models'] if m['key'] == key)
            old_record.update(output=str(output_path.relative_to(ROOT)), output_sha256=sha256(output_path), **{k:metadata[k] for k in ['validation','near_tie_rechecks','efficiency_error','score_zero_denominators','elapsed_seconds']})
            atomic_json(manifest_path, manifest)
            progress_path.unlink(missing_ok=True)
            print(f'COMPLETE {key}: {output_path.name}', flush=True)
    print('ALL 30 MODELS COMPLETE', flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
