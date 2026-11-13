"""Statistics with permutation rows as the only Monte Carlo replicates."""
import numpy as np

TASKS = ['overall'] + [f'digit_{d}' for d in range(10)]
BUDGETS = [100, 250, 500, 750, 1000]


def distributed_computation(contributions):
    """Original D, vectorized over leading dimensions; neurons on last axis."""
    squared = np.asarray(contributions, dtype=np.float64) ** 2
    total = squared.sum(axis=-1, keepdims=True)
    p = np.divide(squared, total, out=np.zeros_like(squared), where=total != 0)
    p = np.where(p == 0, 1e-10, p)
    entropy = -(p * np.log(p)).sum(axis=-1)
    d = np.clip(entropy / np.log(squared.shape[-1]), 0, 1)
    return np.where(total[..., 0] == 0, 0., d)


def all_d(mean_contributions):
    """Input (..., neurons, digits); average signed digit contributions first."""
    overall = distributed_computation(mean_contributions.mean(axis=-1))
    digits = distributed_computation(np.swapaxes(mean_contributions, -1, -2))
    return np.concatenate([overall[..., None], digits], axis=-1)


def score_counts(correct_counts, predicted_counts):
    # evaluation.evaluate_model passes (predictions, labels) to a function whose
    # argument names are (targets, preds). Preserve its confusion-matrix rows.
    return correct_counts.astype(np.float64) / (predicted_counts.astype(np.float64) + 1e-6)


def marginals_from_counts(correct_counts, predicted_counts, permutations):
    scores = score_counts(correct_counts, predicted_counts)
    removal_differences = scores[:, :-1] - scores[:, 1:]
    marginals = np.empty_like(removal_differences)
    np.put_along_axis(marginals, permutations[:, :, None], removal_differences, axis=1)
    return marginals


def whole_permutation_bootstrap(marginals, seed, resamples=10000, chunk_size=250):
    """Multinomial counts equal resampling row indices, shared by every entry."""
    rng = np.random.default_rng(seed)
    n, neurons, digits = marginals.shape
    flattened = np.ascontiguousarray(marginals.reshape(n, -1), dtype=np.float64)
    result = np.empty((resamples, 1 + digits), dtype=np.float64)
    for start in range(0, resamples, chunk_size):
        size = min(chunk_size, resamples - start)
        weights = rng.multinomial(n, np.full(n, 1 / n), size=size).astype(np.float64)
        means = (weights @ flattened / n).reshape(size, neurons, digits)
        result[start:start + size] = all_d(means)
    return result
