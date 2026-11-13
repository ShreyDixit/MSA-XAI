"""Known-answer checks for the nonlinear statistic and resampling unit."""
import numpy as np
from .statistics import all_d, distributed_computation, marginals_from_counts, score_counts, whole_permutation_bootstrap


def original_d(c):
    s = c ** 2
    total = np.sum(s)
    if total == 0:
        return 0.
    p = np.where(s / total == 0, 1e-10, s / total)
    return max(0., min(1., -np.sum(p*np.log(p)) / np.log(len(c))))


def test_original_formula():
    rng = np.random.default_rng(31)
    for c in [np.zeros(200), np.ones(200), np.eye(200)[0], rng.normal(size=200)]:
        assert distributed_computation(c) == original_d(c)


def test_average_signed_tasks_before_d():
    means = np.array([[1., -1.], [1., 1.], [0., 0.]])
    actual = all_d(means)
    assert actual[0] == original_d(means.mean(-1))
    assert actual[0] != np.mean(actual[1:])


def test_permutation_assignment_and_efficiency():
    permutations = np.array([[2, 0, 1], [1, 2, 0]])
    correct = np.array([[[9, 7], [7, 6], [4, 5], [0, 0]], [[9, 7], [6, 5], [5, 2], [0, 0]]])
    predicted = np.array([[[10, 10], [11, 9], [12, 8], [20, 0]], [[10, 10], [12, 8], [13, 7], [20, 0]]])
    marginal = marginals_from_counts(correct, predicted, permutations)
    scores = score_counts(correct, predicted)
    for r in range(2):
        for k, neuron in enumerate(permutations[r]):
            np.testing.assert_array_equal(marginal[r, neuron], scores[r, k]-scores[r, k+1])
    np.testing.assert_allclose(marginal.sum(1), scores[:, 0]-scores[:, -1], atol=1e-15)
    assert scores[0, -1, 1] == 0


def test_bootstrap_matches_joint_multinomial_resampling():
    rng = np.random.default_rng(3)
    # Shared and opposing components make resampling neurons/tasks separately wrong.
    rows = rng.normal(size=(12, 3, 2))
    rows[:, 1, 1] = -rows[:, 0, 0]
    actual = whole_permutation_bootstrap(rows, seed=72, resamples=8, chunk_size=8)
    weights = np.random.default_rng(72).multinomial(12, np.full(12, 1/12), size=8)
    expected = np.stack([all_d(np.repeat(rows, w, axis=0).mean(0)) for w in weights])
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_original_argument_order_gives_precision():
    # Predictions [0,0,0,1], labels [0,0,1,1]: precision=[2/3,1], recall=[1,1/2].
    actual = score_counts(np.array([2,1]), np.array([3,1]))
    np.testing.assert_allclose(actual, np.array([2/(3+1e-6),1/(1+1e-6)]))
