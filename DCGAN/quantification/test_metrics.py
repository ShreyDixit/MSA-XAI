"""Known-answer checks for definitions that can materially change conclusions."""

import itertools
import pickle

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal

from quantification.metrics import (
    bootstrap_mean, capture, mismatched_capture, shuffled_capture,
    spectral_centroid, top_mask,
)
from quantification.storage import load_modes


def test_top_twenty_percent_has_exact_size_and_seeded_ties():
    flat = np.ones((128, 128))
    a = top_mask(flat, .2, np.random.default_rng(2810))
    b = top_mask(flat, .2, np.random.default_rng(2810))
    assert a.sum() == 3277
    assert_array_equal(a, b)
    # All the strongest pixels must be included, independently of tied pixels.
    flat[:2] = 5
    c = top_mask(flat, .2, np.random.default_rng(2810))
    assert c[:2].all()


def test_capture_concentrated_diffuse_and_zero_change():
    mask = np.array([[True, False], [False, False]])
    assert capture(mask.astype(float), mask) == 1
    assert capture(np.ones((2, 2)), mask) == .25
    assert np.isnan(capture(np.zeros((2, 2)), mask))


def test_spatial_shuffle_has_exact_uniform_expectation():
    change = np.array([[7., 1.], [2., 3.]])
    mask = np.array([[True, False], [False, True]])
    permutations = np.array(list(itertools.permutations(range(4))))
    values = shuffled_capture(change, mask, permutations)
    assert_allclose(values.mean(), .5, atol=1e-15)
    assert np.std(values) > 0


def test_mismatched_aggregate_matches_every_other_filter_explicitly():
    rng = np.random.default_rng(9)
    masks = np.array([top_mask(rng.random((8, 8)), .2, rng) for _ in range(7)])
    change = rng.random((8, 8))
    counts = masks.sum(axis=0).astype(np.uint16)
    for target in range(len(masks)):
        brute = np.mean([capture(change, masks[other]) for other in range(len(masks)) if target != other])
        assert_allclose(mismatched_capture(change, masks[target], counts, len(masks)), brute, atol=1e-15)


def test_signed_sinusoid_has_known_centroid_independent_of_dc_and_scale():
    n = 64
    wave = np.broadcast_to(np.sin(2 * np.pi * 4 * np.arange(n) / n), (n, n))
    maps = np.zeros((3, 3, n, n))
    maps[0, 0] = wave
    maps[1, 0] = 17 * wave + 23
    maps[2, 0] = np.broadcast_to(np.sin(2 * np.pi * 16 * np.arange(n) / n), (n, n))
    centroids, energy = spectral_centroid(maps)
    assert_allclose(centroids, np.array([4, 4, 16]) / n / np.sqrt(.5), atol=1e-13)
    assert energy[1] > energy[0]


def test_rgb_power_combination_avoids_signed_channel_cancellation():
    n = 32
    wave = np.broadcast_to(np.sin(2 * np.pi * 3 * np.arange(n) / n), (n, n))
    maps = np.zeros((1, 3, n, n))
    maps[0, 0], maps[0, 1] = wave, -wave
    centroid, energy = spectral_centroid(maps)
    assert_allclose(centroid, [3 / n / np.sqrt(.5)], atol=1e-13)
    assert energy[0] > 0


def test_nyquist_corner_normalizes_to_one_and_constants_have_zero_energy():
    grid = np.indices((32, 32)).sum(axis=0)
    maps = np.zeros((2, 3, 32, 32))
    maps[0, 0] = (-1.)**grid
    maps[1] = 0.1
    centroids, energy = spectral_centroid(maps)
    assert_allclose(centroids[0], 1)
    assert np.isnan(centroids[1])
    assert energy[1] == 0


def test_bootstrap_omits_undefined_images():
    mean, lo, hi = bootstrap_mean([2, 2, np.nan], 1000, np.random.default_rng(3))
    assert (mean, lo, hi) == (2, 2, 2)


def test_mapped_loader_matches_standard_pickle_for_multimegabyte_modes(tmp_path):
    from msapy.datastructures import ShapleyModeND
    array = np.arange(300000, dtype=np.float64).reshape(100000, 3)
    modes = ShapleyModeND(pd.DataFrame(array, columns=[(0, 0), (0, 1), (1, 0)]), (1, 1, 250, 400))
    path = tmp_path / "modes.pkl"
    with path.open("wb") as f:
        pickle.dump(modes, f, protocol=5)
    standard = pickle.loads(path.read_bytes())
    mapped, reader = load_modes(path)
    assert list(mapped.columns) == list(standard.columns)
    assert mapped._shape == standard._shape
    assert_array_equal(mapped.values, standard.values)
    assert_array_equal(mapped.get_shapley_mode((0, 1)), standard.get_shapley_mode((0, 1)))
    assert not mapped.values.flags.writeable
