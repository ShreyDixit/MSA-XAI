"""Array-only definitions of the spatial agreement and spatial scale metrics."""

import numpy as np


def top_mask(magnitude, fraction, rng):
    """Select exactly ceil(fraction * H * W) pixels; randomize boundary ties."""
    flat = np.asarray(magnitude).ravel()
    if not 0 < fraction < 1 or not np.isfinite(flat).all():
        raise ValueError("Require a finite magnitude map and fraction in (0, 1).")
    k = int(np.ceil(fraction * flat.size))
    order = rng.permutation(flat.size)
    chosen = order[np.argpartition(-flat[order], k - 1)[:k]]
    mask = np.zeros(flat.size, dtype=bool)
    mask[chosen] = True
    return mask.reshape(magnitude.shape)


def capture(change, mask):
    """Fraction of absolute change captured; zero-change maps are undefined."""
    total = np.sum(change, dtype=np.float64)
    return np.sum(change[mask], dtype=np.float64) / total if total > 0 else np.nan


def shuffled_capture(change, mask, permutations):
    """Each permutation spatially shuffles the selected pixels of the map."""
    total = np.sum(change, dtype=np.float64)
    if total == 0:
        return np.full(len(permutations), np.nan)
    selected = np.flatnonzero(mask)
    return np.sum(change.ravel()[permutations[:, selected]], axis=1, dtype=np.float64) / total


def mismatched_capture(change, own_mask, layer_mask_count, eligible_references):
    """Exact mean capture using every eligible OTHER filter in the same layer."""
    total = np.sum(change, dtype=np.float64)
    if total == 0 or eligible_references < 2:
        return np.nan
    other_counts = layer_mask_count.astype(np.float64) - own_mask
    return np.sum(change * other_counts) / ((eligible_references - 1) * total)


def spectral_centroid(signed_maps, window=False):
    """Maps have shape (N,C,H,W). Demean EACH channel, sum channel FFT powers.

    Radial frequency is in cycles/pixel, divided by sqrt(0.5**2 + 0.5**2),
    the 2-D Nyquist corner. The full FFT counts both sides of the spectrum.
    Zero non-DC power gives NaN. Energy is spatial energy (Parseval scaling).
    Optional Hann taper is a boundary sensitivity analysis, not the primary metric.
    """
    maps = np.asarray(signed_maps, dtype=np.float64)
    centered = maps - maps.mean(axis=(-2, -1), keepdims=True)
    h, w = centered.shape[-2:]
    if window:
        centered = centered * (np.hanning(h)[:, None] * np.hanning(w)[None, :])
    power = np.square(np.abs(np.fft.fft2(centered, axes=(-2, -1)))).sum(axis=1)
    power[:, 0, 0] = 0  # explicitly exclude DC, including mean-roundoff residuals
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    radius = np.sqrt(fy**2 + fx**2) / np.sqrt(0.5**2 + 0.5**2)
    energy = power.sum(axis=(-2, -1))
    centroid = np.divide(
        (power * radius).sum(axis=(-2, -1)), energy,
        out=np.full(len(maps), np.nan), where=energy > 0,
    )
    return centroid, energy / (h * w)


def bootstrap_mean(values, draws, rng):
    """Percentile 95% interval over independent image-level summaries."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    means = values[rng.integers(0, len(values), (draws, len(values)))].mean(axis=1)
    lo, hi = np.quantile(means, (0.025, 0.975))
    return values.mean(), lo, hi
