#!/usr/bin/env python3
"""Reproduce Options 2 and 3 for the larger saved CelebA generator.

Run from the repository root: python DCGAN/quantify.py --device cuda:0
See quantification/README.md for definitions, dependencies, and resumable stages.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dcgan-quantification-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

from quantification.metrics import (
    bootstrap_mean, capture, mismatched_capture, shuffled_capture,
    spectral_centroid, top_mask,
)
from quantification.storage import fingerprint, load_modes, sha256

BASE = Path(__file__).resolve().parent
WIDTHS = (1024, 512, 256, 128, 64, 3)
UNITS = [(layer, unit) for layer, n in enumerate(WIDTHS) for unit in range(n)]


def log(message):
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {message}", flush=True)


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def csv(frame, path):
    temp = path.with_suffix(".tmp")
    frame.to_csv(temp, index=False, float_format="%.12g")
    temp.replace(path)


def rng_for(seed, *keys):
    return np.random.default_rng(np.random.SeedSequence([seed, *keys]))


def model_and_latents(args):
    import torch
    from model import Generator
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    # Match the historical notebook defaults. Enforcing deterministic cuDNN
    # selects different convolution algorithms and changes reconstructed pixels.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    generator = Generator(100, 64, 3, 6).to(args.device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    generator.eval()
    latents = torch.load(args.latents, map_location="cpu", weights_only=True)
    if tuple(latents.shape) != (32, 100, 1, 1) or not torch.isfinite(latents).all():
        raise ValueError("Expected all 32 original finite latent vectors (32,100,1,1).")
    return generator, latents.to(args.device)


def infer(generator, latents, args, lesion=None):
    import torch
    with torch.inference_mode():
        return np.concatenate([
            generator(latents[b:b + args.batch_size], lesion).cpu().numpy()
            for b in range(0, len(latents), args.batch_size)
        ])


def initialize(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    log("Hashing input artifacts and analysis source")
    identity = {
        "inputs": {key: fingerprint(getattr(args, key)) for key in ("checkpoint", "latents", "modes")},
        "source_sha256": {str(p.relative_to(BASE)): sha256(p) for p in (
            BASE / "model.py", Path(__file__), BASE / "quantification/metrics.py",
            BASE / "quantification/storage.py",
        )},
        "settings": {k: getattr(args, k) for k in (
            "device", "batch_size", "seed", "fraction", "shuffles", "bootstrap", "threads",
        )},
    }
    cache_manifest = args.cache / "identity.json"
    if cache_manifest.exists():
        if json.loads(cache_manifest.read_text()) != identity:
            raise ValueError("Cache identity differs. Use a new --cache and --output directory.")
    else:
        if any(args.cache.iterdir()):
            raise ValueError("Nonempty cache has no identity; use a fresh cache directory.")
        write_json(cache_manifest, identity)
    modes, reader = load_modes(args.modes)
    if tuple(modes._shape) != (32, 3, 128, 128) or list(modes.columns) != UNITS:
        raise ValueError("Mode shapes or ordered (layer,filter) mapping do not match the larger model.")
    if modes.shape != (32 * 3 * 128 * 128, len(UNITS)):
        raise ValueError("Unexpected flattened mode shape")
    manifest = {
        **identity, "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {name: importlib.metadata.version(name) for name in (
            "torch", "torchvision", "numpy", "pandas", "scipy", "matplotlib", "msapy",
        )},
        "image_count": 32, "mode_shape": list(modes._shape),
        "layer_widths": list(WIDTHS), "historical_estimates_only": True,
        "historical_permutations": {"count": 3000, "source": "CelebA GAN.ipynb cell 2; not stored in pickle"},
        "lesion": "Zero the chosen output channel after the complete layer block (ConvTranspose2d, BatchNorm2d, ReLU); final block is ConvTranspose2d,Sigmoid.",
        "mode_game": "All 1984 hidden filters AND three output RGB channels are players, as in the notebook.",
        "precision": {"inference_dtype": "float32", "mode_dtype": str(modes.values.dtype),
                      "cudnn_allow_tf32": True, "cudnn_deterministic": False,
                      "cudnn_benchmark": False, "matmul_allow_tf32": False},
    }
    write_json(args.output / "manifest.json", manifest)
    csv(pd.DataFrame([{"mode_column": i, "layer": l, "filter": u,
                       "role": "hidden" if l < 5 else "output_rgb",
                       "channel": "RGB"[u] if l == 5 else "",
                       "activation_resolution": 4 * 2**l}
                      for i, (l, u) in enumerate(UNITS)]), args.output / "unit_mapping.csv")
    return modes, reader


def validate(args, modes, generator, latents):
    log("Validating summed modes against all 32 checkpoint outputs")
    baseline = infer(generator, latents, args)
    repeated = infer(generator, latents, args)
    empty = infer(generator, latents, args, {l: list(range(n)) for l, n in enumerate(WIDTHS)})
    total = modes.values.sum(axis=1, dtype=np.float64).reshape(modes._shape)
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite summed modes")
    residual = total - (baseline - empty)
    rows = []
    for image in range(32):
        distances = np.mean((baseline.astype(np.float64) - empty - total[image])**2, axis=(1, 2, 3))
        rows.append({"image": image, "mae": np.abs(residual[image]).mean(),
                     "max_abs_error": np.abs(residual[image]).max(),
                     "relative_l2": np.linalg.norm(residual[image]) / np.linalg.norm(baseline[image] - empty[image]),
                     "nearest_reconstructed_image": int(np.argmin(distances)),
                     "repeat_max_abs_error": np.abs(baseline[image] - repeated[image]).max()})
    qc = pd.DataFrame(rows)
    csv(qc, args.output / "compatibility_by_image.csv")
    passed = bool((qc.relative_l2 < 1e-5).all() and (qc.max_abs_error < 1e-5).all()
                  and np.array_equal(qc.nearest_reconstructed_image, np.arange(32))
                  and np.max(np.abs(empty)) == 0)
    import torch
    result = {"passed": passed, "maximum_absolute_residual": float(np.max(np.abs(residual))),
              "mean_absolute_residual": float(np.mean(np.abs(residual))),
              "max_repeat_difference": float(np.max(np.abs(baseline - repeated))),
              "absolute_tolerance": 1e-5, "relative_l2_tolerance": 1e-5,
              "all_lesioned_output_max_abs": float(np.max(np.abs(empty))),
              "image_order_matches": bool(np.array_equal(qc.nearest_reconstructed_image, np.arange(32))),
              "historical_provenance": "No historical artifact hashes were stored; compatibility is supported by full-image Shapley efficiency and ordered unit/latent checks, not a historical hash attestation.",
              "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "CPU",
              "cudnn_version": torch.backends.cudnn.version()}
    write_json(args.output / "compatibility.json", result)
    if not passed:
        raise ValueError("Historical modes failed compatibility checks; no mixed or new estimates were generated.")
    np.save(args.cache / "baseline.npy", baseline)
    np.save(args.cache / "mode_sum.npy", total)
    np.save(args.cache / "latents.npy", latents.cpu().numpy())
    csv(pd.DataFrame(latents.cpu().numpy().reshape(32, 100)).rename_axis("image").reset_index(),
        args.output / "latent_vectors.csv")
    log(f"Compatibility passed: max residual {result['maximum_absolute_residual']:.3g}")


def cached_array(path, shape, dtype):
    if path.exists():
        result = np.load(path, mmap_mode="r+")
        if result.shape != shape or result.dtype != np.dtype(dtype):
            raise ValueError(f"Incompatible cached array: {path}")
        return result
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def ablations(args, generator, latents):
    log("Generating/resuming all single-hidden-filter ablations")
    baseline = np.load(args.cache / "baseline.npy")
    started = time.monotonic()
    rows = []
    for layer, width in enumerate(WIDTHS[:-1]):
        outputs = cached_array(args.cache / f"ablations_layer_{layer}.npy", (width, 32, 3, 128, 128), "float32")
        done = cached_array(args.cache / f"ablations_layer_{layer}_done.npy", (width,), "bool")
        for unit in range(width):
            if not done[unit]:
                value = infer(generator, latents, args, {layer: [unit]})
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite ablation {layer, unit}")
                outputs[unit] = value
                # Commit output before its completion marker, permitting safe resume.
                outputs.flush()
                done[unit] = True
                done.flush()
            if unit % 128 == 0 or unit == width - 1:
                log(f"Ablations layer {layer}: {unit + 1}/{width}; elapsed {time.monotonic() - started:.0f}s")
        outputs.flush()
        done.flush()
        # Check cached intervention semantics against the public model API.
        for unit in sorted({0, width // 2, width - 1}):
            check = infer(generator, latents, args, {layer: [unit]})
            error = float(np.max(np.abs(outputs[unit] - check)))
            rows.append({"layer": layer, "filter": unit, "repeat_max_abs_error": error,
                         "passed": error < 1e-5})
        log(f"Layer {layer} cache complete; baseline range {baseline.min():.4g}–{baseline.max():.4g}")
    checks = pd.DataFrame(rows)
    csv(checks, args.output / "ablation_repeat_checks.csv")
    if not checks.passed.all():
        raise ValueError("Ablation repeat checks failed")


def mode_metrics(args, modes):
    log("Computing top-pixel masks and signed-map spectral centroids")
    rows = []
    started = time.monotonic()
    for layer, width in enumerate(WIDTHS):
        masks = cached_array(args.cache / f"masks_layer_{layer}.npy", (width, 32, 2048), "uint8")
        counts = np.zeros((32, 128, 128), dtype=np.uint16)
        eligible = np.zeros(32, dtype=np.int32)
        for unit in range(width):
            maps = modes.get_shapley_mode((layer, unit))
            if not np.isfinite(maps).all():
                raise ValueError(f"Nonfinite modes at {layer, unit}")
            magnitude = np.abs(maps).mean(axis=1)
            centroid, energy = spectral_centroid(maps)
            hann_centroid, hann_energy = spectral_centroid(maps, window=True)
            for image in range(32):
                nonzero = bool(np.any(magnitude[image] > 0))
                mask = top_mask(magnitude[image], args.fraction, rng_for(args.seed, 1, layer, unit, image))
                masks[unit, image] = np.packbits(mask.ravel())
                if nonzero:
                    counts[image] += mask
                    eligible[image] += 1
                rows.append({"image": image, "layer": layer, "filter": unit,
                             "role": "hidden" if layer < 5 else "output_rgb",
                             "channel": "RGB"[unit] if layer == 5 else "",
                             "attribution_abs_total": magnitude[image].sum(),
                             "zero_attribution": not nonzero,
                             "constant_magnitude": bool(np.ptp(magnitude[image]) == 0),
                             "spectral_centroid": centroid[image], "non_dc_energy": energy[image],
                             "zero_energy": bool(energy[image] == 0),
                             "hann_centroid": hann_centroid[image], "hann_energy": hann_energy[image]})
            if unit % 128 == 0 or unit == width - 1:
                log(f"Modes layer {layer}: {unit + 1}/{width}; elapsed {time.monotonic() - started:.0f}s")
        masks.flush()
        np.save(args.cache / f"mask_counts_layer_{layer}.npy", counts)
        np.save(args.cache / f"mask_eligible_layer_{layer}.npy", eligible)
    csv(pd.DataFrame(rows), args.output / "spectral_by_filter_image.csv")


def agreement(args):
    log(f"Computing spatial capture, {args.shuffles} shuffles, and all mismatched-filter controls")
    baseline = np.load(args.cache / "baseline.npy")
    spectral = pd.read_csv(args.output / "spectral_by_filter_image.csv").set_index(["layer", "filter", "image"])
    # An independent bank for each image, shared across its filters and layers.
    # This preserves the dependence of null summaries within an image.
    permutations = [np.array([rng_for(args.seed, 2, image, rep).permutation(128 * 128)
                              for rep in range(args.shuffles)], dtype=np.int32) for image in range(32)]
    started = time.monotonic()
    for layer, width in enumerate(WIDTHS[:-1]):
        path = args.output / f"agreement_layer_{layer}.csv"
        if path.exists():
            log(f"Using completed layer {layer} agreement CSV")
            continue
        outputs = np.load(args.cache / f"ablations_layer_{layer}.npy", mmap_mode="r")
        if not np.load(args.cache / f"ablations_layer_{layer}_done.npy").all():
            raise ValueError(f"Incomplete ablation cache for layer {layer}")
        masks = np.load(args.cache / f"masks_layer_{layer}.npy", mmap_mode="r")
        counts = np.load(args.cache / f"mask_counts_layer_{layer}.npy")
        eligible = np.load(args.cache / f"mask_eligible_layer_{layer}.npy")
        rows = []
        for unit in range(width):
            difference = np.abs(baseline.astype(np.float64) - outputs[unit]).mean(axis=1)
            for image in range(32):
                mask = np.unpackbits(masks[unit, image]).reshape(128, 128).astype(bool)
                change = difference[image]
                total = change.sum()
                zero_attr = bool(spectral.loc[(layer, unit, image), "zero_attribution"])
                valid = total > 0 and not zero_attr
                score = capture(change, mask) if valid else np.nan
                shuffles = shuffled_capture(change, mask, permutations[image]) if valid else np.full(args.shuffles, np.nan)
                mismatch = mismatched_capture(change, mask, counts[image], int(eligible[image])) if valid else np.nan
                reference = mask.sum() / mask.size
                rows.append({"image": image, "layer": layer, "filter": unit,
                             "change_abs_total": total, "change_abs_mean": total / mask.size,
                             "zero_change": total == 0, "near_zero_change": 0 < total / mask.size < 1e-6,
                             "zero_attribution": zero_attr, "valid_agreement": valid,
                             "top_pixels": int(mask.sum()), "top_fraction_actual": reference,
                             "capture": score, "shuffle_exact_mean": reference if valid else np.nan,
                             "shuffle_mc_mean": np.mean(shuffles) if valid else np.nan,
                             "shuffle_mc_sd": np.std(shuffles, ddof=1) if valid else np.nan,
                             "shuffle_mc_q025": np.quantile(shuffles, .025) if valid else np.nan,
                             "shuffle_mc_q975": np.quantile(shuffles, .975) if valid else np.nan,
                             "mismatched_mean": mismatch,
                             "mismatched_reference_count": max(int(eligible[image]) - (not zero_attr), 0),
                             "gain_vs_shuffle": score - reference,
                             "gain_vs_mismatched": score - mismatch})
            if unit % 128 == 0 or unit == width - 1:
                log(f"Agreement layer {layer}: {unit + 1}/{width}; elapsed {time.monotonic() - started:.0f}s")
        csv(pd.DataFrame(rows), path)
    csv(pd.concat([pd.read_csv(args.output / f"agreement_layer_{l}.csv") for l in range(5)], ignore_index=True),
        args.output / "agreement_by_filter_image.csv")


def summarize(args):
    spatial = pd.read_csv(args.output / "agreement_by_filter_image.csv")
    spectral = pd.read_csv(args.output / "spectral_by_filter_image.csv")
    spatial_metrics = ["capture", "shuffle_mc_mean", "shuffle_exact_mean", "mismatched_mean",
                       "gain_vs_shuffle", "gain_vs_mismatched"]
    image_rows = []
    for (image, layer), group in spatial.groupby(["image", "layer"]):
        image_rows.append({"image": image, "layer": layer, "filters": len(group),
                           "valid_filters": int(group.valid_agreement.sum()),
                           "zero_change_filters": int(group.zero_change.sum()),
                           "near_zero_change_filters": int(group.near_zero_change.sum()),
                           "zero_attribution_filters": int(group.zero_attribution.sum()),
                           **{m: group[m].mean() for m in spatial_metrics}})
    by_image = pd.DataFrame(image_rows)
    csv(by_image, args.output / "agreement_by_image_layer.csv")
    rows = []
    for layer, group in by_image.groupby("layer"):
        for j, metric in enumerate(spatial_metrics):
            mean, lo, hi = bootstrap_mean(group[metric], args.bootstrap, rng_for(args.seed, 3, int(layer), j))
            rows.append({"layer": layer, "metric": metric, "mean": mean,
                         "ci025": lo, "ci975": hi, "images": int(group[metric].notna().sum())})
    layer_agreement = pd.DataFrame(rows)
    csv(layer_agreement, args.output / "agreement_by_layer.csv")
    energy_images = []
    for (image, layer), group in spectral.groupby(["image", "layer"]):
        energy_images.append({"image": image, "layer": layer, "role": group.role.iloc[0],
                              "filters": len(group), "valid_filters": int((~group.zero_energy).sum()),
                              "zero_energy_filters": int(group.zero_energy.sum()),
                              "spectral_centroid": group.spectral_centroid.mean(),
                              "hann_centroid": group.hann_centroid.mean()})
    energy_images = pd.DataFrame(energy_images)
    csv(energy_images, args.output / "spectral_by_image_layer.csv")
    rows = []
    for layer, group in energy_images.groupby("layer"):
        for j, metric in enumerate(("spectral_centroid", "hann_centroid")):
            mean, lo, hi = bootstrap_mean(group[metric], args.bootstrap, rng_for(args.seed, 4, int(layer), j))
            rows.append({"layer": layer, "role": group.role.iloc[0], "metric": metric,
                         "mean": mean, "ci025": lo, "ci975": hi,
                         "images": int(group[metric].notna().sum()),
                         "zero_energy_maps": int(group.zero_energy_filters.sum())})
    layer_spectral = pd.DataFrame(rows)
    csv(layer_spectral, args.output / "spectral_by_layer.csv")
    rgb = spectral[spectral.role == "output_rgb"]
    csv(rgb, args.output / "spectral_output_rgb.csv")
    rgb_rows = []
    for channel, group in rgb.groupby("channel"):
        mean, lo, hi = bootstrap_mean(group.spectral_centroid, args.bootstrap, rng_for(args.seed, 5, "RGB".index(channel)))
        rgb_rows.append({"channel": channel, "mean": mean, "ci025": lo, "ci975": hi,
                         "images": int(group.spectral_centroid.notna().sum()), "zero_energy_maps": int(group.zero_energy.sum())})
    csv(pd.DataFrame(rgb_rows), args.output / "spectral_output_rgb_summary.csv")
    # Describe layer trends within each image, avoiding pseudoreplication over units.
    trend_rows = []
    from scipy.stats import spearmanr
    for image, group in energy_images[energy_images.layer < 5].groupby("image"):
        group = group.sort_values("layer")
        values = group.spectral_centroid.to_numpy()
        trend_rows.append({"image": image, "spearman_layer_centroid": spearmanr(group.layer, values).statistic,
                           "last_minus_first": values[-1] - values[0],
                           "strictly_increasing": bool(np.all(np.diff(values) > 0)),
                           "hann_last_minus_first": group.hann_centroid.iloc[-1] - group.hann_centroid.iloc[0]})
    trends = pd.DataFrame(trend_rows)
    csv(trends, args.output / "spectral_trend_by_image.csv")
    trend_summary = []
    for j, name in enumerate(("spearman_layer_centroid", "last_minus_first", "hann_last_minus_first")):
        mean, lo, hi = bootstrap_mean(trends[name], args.bootstrap, rng_for(args.seed, 6, j))
        trend_summary.append({"metric": name, "mean": mean, "ci025": lo, "ci975": hi, "images": len(trends)})
    csv(pd.DataFrame(trend_summary), args.output / "spectral_trend_summary.csv")
    exclusions = spatial.groupby("layer").agg(
        maps=("image", "size"), zero_change=("zero_change", "sum"),
        near_zero_change=("near_zero_change", "sum"), zero_attribution=("zero_attribution", "sum"),
        valid=("valid_agreement", "sum")).reset_index()
    csv(exclusions, args.output / "agreement_exclusions.csv")
    return spatial, spectral, by_image, energy_images, layer_agreement, layer_spectral, trends


def figures(args, spatial, spectral, by_image, energy_images, layer_agreement, layer_spectral, trends):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    target = args.output / "figures"
    target.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            fig.savefig(target / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    for offset, metric, label, color in ((-.18, "capture", "Matched filter", "#247ba0"),
                                         (0, "mismatched_mean", "Other filters, same layer", "#e18335"),
                                         (.18, "shuffle_mc_mean", "Spatially shuffled", "#777777")):
        table = layer_agreement[layer_agreement.metric == metric].sort_values("layer")
        ax.errorbar(table.layer + offset, table["mean"],
                    yerr=np.maximum(0, np.vstack([table["mean"] - table.ci025, table.ci975 - table["mean"]])),
                    fmt="o", capsize=3, label=label, color=color)
    ax.axhline(np.ceil(args.fraction * 16384) / 16384, color="grey", linestyle="--", linewidth=1)
    ax.set(xticks=range(5), xlabel="Hidden layer (0 = earliest)", ylabel="Fraction of image-change magnitude captured",
           title="Top-20% attribution pixels: single-filter intervention agreement")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=9)
    save(fig, "spatial_agreement")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, ("gain_vs_shuffle", "gain_vs_mismatched"), ("Gain over spatial shuffle", "Gain over other filters")):
        data = [by_image.loc[by_image.layer == l, metric].dropna().to_numpy() for l in range(5)]
        ax.boxplot(data, positions=range(5), showfliers=False)
        for image, group in by_image.groupby("image"):
            ax.plot(group.layer, group[metric], color="#247ba0", alpha=.15, linewidth=.6)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set(xlabel="Hidden layer", ylabel="Capture difference", title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "spatial_agreement_by_image")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hidden = spectral[spectral.role == "hidden"]
    data = [hidden.loc[hidden.layer == l, "spectral_centroid"].dropna().to_numpy() for l in range(5)]
    axes[0].violinplot(data, positions=range(5), showmedians=True)
    axes[0].set(xticks=range(5), xlabel="Hidden layer", ylabel="Normalized spectral centroid",
                title="All hidden-filter modes, all images")
    for image, group in energy_images[energy_images.layer < 5].groupby("image"):
        axes[1].plot(group.layer, group.spectral_centroid, color="#247ba0", alpha=.2, linewidth=.7)
    for metric, label, color in (("spectral_centroid", "Primary", "#247ba0"), ("hann_centroid", "Hann sensitivity", "#e18335")):
        table = layer_spectral[(layer_spectral.metric == metric) & (layer_spectral.layer < 5)].sort_values("layer")
        axes[1].errorbar(table.layer, table["mean"],
                         yerr=np.maximum(0, np.vstack([table["mean"] - table.ci025, table.ci975 - table["mean"]])),
                         fmt="o-", color=color, capsize=3, label=label)
    axes[1].set(xticks=range(5), xlabel="Hidden layer", ylabel="Mean normalized spectral centroid",
                title="Image-level means and 95% bootstrap intervals")
    axes[1].legend()
    save(fig, "layer_spatial_scale")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    rgb = spectral[spectral.role == "output_rgb"]
    ax.boxplot([rgb.loc[rgb.channel == c, "spectral_centroid"].dropna() for c in "RGB"], labels=list("RGB"))
    ax.set(xlabel="Output-channel player", ylabel="Normalized spectral centroid",
           title="Final RGB output modes (separate from hidden layers)")
    save(fig, "output_rgb_spatial_scale")

    # Fixed indices are specified independently of visual appearance or semantics.
    modes, reader = load_modes(args.modes)
    baseline = np.load(args.cache / "baseline.npy")
    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    for layer in range(5):
        maps = modes.get_shapley_mode((layer, 0))[0]
        magnitude = np.abs(maps).mean(axis=0)
        outputs = np.load(args.cache / f"ablations_layer_{layer}.npy", mmap_mode="r")
        change = np.abs(baseline[0].astype(np.float64) - outputs[0, 0]).mean(axis=0)
        mask = np.unpackbits(np.load(args.cache / f"masks_layer_{layer}.npy", mmap_mode="r")[0, 0]).reshape(128, 128)
        axes[layer, 0].imshow(baseline[0].transpose(1, 2, 0))
        axes[layer, 1].imshow(magnitude, cmap="magma")
        axes[layer, 2].imshow(change, cmap="magma")
        axes[layer, 3].imshow(change, cmap="magma")
        axes[layer, 3].contour(mask, levels=[.5], colors="cyan", linewidths=.5)
        axes[layer, 0].set_ylabel(f"Layer {layer}, filter 0")
        for ax in axes[layer]:
            ax.set_xticks([])
            ax.set_yticks([])
    for ax, title in zip(axes[0], ("Image 0", "Attribution magnitude", "Single-filter change", "Top-20% mask on change")):
        ax.set_title(title, fontsize=10)
    save(fig, "fixed_spatial_examples")


def report(args, data):
    spatial, spectral, by_image, energy_images, layer_agreement, layer_spectral, trends = data
    manifest = json.loads((args.output / "manifest.json").read_text())
    qc = json.loads((args.output / "compatibility.json").read_text())
    def estimate(frame, layer, metric):
        row = frame[(frame.layer == layer) & (frame.metric == metric)].iloc[0]
        return f"{row['mean']:.4f} [{row.ci025:.4f}, {row.ci975:.4f}]"
    lines = ["# DCGAN quantification: larger CelebA generator", "",
             "## Dataset and compatibility", "",
             f"One historical dataset: {len(spectral)} signed, unnormalized filter–image modes, "
             "32 fixed latent vectors, 1,984 hidden filters (layers 0–4: 1,024/512/256/128/64), "
             "and three final RGB players (layer 5). Each mode is 3×128×128. "
             "The historical Shapley game included both hidden and output units; output-channel gating "
             "is therefore part of the coalition context for the hidden modes. No new Shapley estimates were mixed in.", "",
             f"Checkpoint `{manifest['inputs']['checkpoint']['path']}`; SHA-256 "
             f"`{manifest['inputs']['checkpoint']['sha256']}`. Full identities, versions, precision settings, "
             "and perturbation definitions are in `manifest.json`; ordered unit mapping and all latent values are in CSVs.", "",
             f"Summed modes reconstruct the unlesioned checkpoint output minus the fully lesioned output "
             f"with maximum absolute residual {qc['maximum_absolute_residual']:.3g} and mean absolute residual "
             f"{qc['mean_absolute_residual']:.3g}. All 32 images match their own latent indices; fully lesioned output is zero. "
             f"Repeated unlesioned inference differs by at most {qc['max_repeat_difference']:.3g}. "
             "This supports compatibility; the historical run did not store input hashes. "
             "Inference uses the notebook's TF32/cuDNN defaults rather than forcing a different deterministic algorithm.", "",
             "## Option 2: spatial intervention agreement", "",
             "For each hidden filter and image, A(x,y)=mean_RGB(|signed Shapley mode|) and "
             "D(x,y)=mean_RGB(|unlesioned image − single-filter-lesioned image|). Lesions zero one "
             "post-BatchNorm/ReLU channel using `Generator.forward` in evaluation mode, with the same "
             "checkpoint and fixed latents as the modes. Raw outputs are cached separately by layer.", "",
             f"The selected mask contains exactly ceil(0.20×128×128)=3,277 pixels "
             f"({3277/16384:.8%}); boundary ties are broken by seeded random ordering. "
             "Capture is sum(D inside mask)/sum(D). Zero-change cases are undefined and excluded, "
             "with counts reported separately. Zero-attribution cases are also excluded; near-zero "
             "change (positive mean change <10⁻⁶) is flagged but retained in the primary analysis.", "",
             f"Controls: {args.shuffles} seeded uniform spatial permutations of each attribution mask; "
             "their exact expected capture is the selected pixel fraction. The mismatched control is "
             "the exact mean capture over every other eligible filter's attribution mask in the same "
             "layer and image. It is computed from summed masks, without sampling filter pairs. "
             "Uniform permutation destroys spatial structure; the layer control retains layer-specific spatial biases.", "",
             f"Scores are averaged equally across valid filters within each image/layer, then equally across "
             f"images. Brackets below are 95% percentile intervals from {args.bootstrap:,} image bootstrap "
             "resamples; gain intervals use paired differences. They quantify variation over these saved "
             "images, conditional on one checkpoint and the saved Shapley estimates. Pixels and filters "
             "are not treated as independent replicates. No per-filter significance claims are made.", "",
             "| Hidden layer | Matched capture | Gain vs spatial shuffle | Gain vs other filters |",
             "|---|---:|---:|---:|"]
    for layer in range(5):
        lines.append(f"| {layer} | {estimate(layer_agreement, layer, 'capture')} | "
                     f"{estimate(layer_agreement, layer, 'gain_vs_shuffle')} | "
                     f"{estimate(layer_agreement, layer, 'gain_vs_mismatched')} |")
    lines += ["", f"Of {len(spatial):,} hidden filter–image cases, {int(spatial.zero_change.sum()):,} have "
              f"zero change, {int(spatial.zero_attribution.sum()):,} have zero attribution, and "
              f"{int(spatial.near_zero_change.sum()):,} have positive mean change below 10⁻⁶. "
              "See `agreement_exclusions.csv` for layer counts.", "",
              "Agreement describes localization of single-filter effects. Shapley Modes average effects "
              "over coalitions, so perfect agreement with removal from the fully intact network is not "
              "required. Exceeding spatial shuffle alone can reflect generic spatial organization; "
              "the same-layer mismatched comparison addresses filter specificity.", "",
              "## Option 3: layer-wise spatial scale", "",
              "Each signed RGB channel is demeaned over space before a full two-dimensional FFT. "
              "Power is summed across channels (no cancellation across RGB), with DC explicitly removed. "
              "The centroid weights radial frequency sqrt(fx²+fy²), in cycles/pixel, by this combined "
              "power and divides by sqrt(0.5²+0.5²), the 2-D Nyquist corner. The normalized range is "
              "0–1, where larger values indicate finer spatial variation. Zero non-DC-energy maps "
              "receive NaN and are excluded from scale summaries. Modes are not rectified or "
              "display-normalized before this calculation.", "",
              "Primary estimates use no taper. A separable Hann taper applied after demeaning provides "
              "a sensitivity check for image-edge discontinuities; its DC is also discarded. "
              "Distributions show individual filter–image values. Layer means and uncertainty first "
              "average valid filters per image, then bootstrap images as above.", "",
              "| Hidden layer | Spectral centroid | Hann sensitivity centroid |",
              "|---|---:|---:|"]
    for layer in range(5):
        lines.append(f"| {layer} | {estimate(layer_spectral, layer, 'spectral_centroid')} | "
                     f"{estimate(layer_spectral, layer, 'hann_centroid')} |")
    trend_summary = pd.read_csv(args.output / "spectral_trend_summary.csv")
    delta = trend_summary[trend_summary.metric == "last_minus_first"].iloc[0]
    increasing = bool(np.all(np.diff(layer_spectral[(layer_spectral.layer < 5) &
                            (layer_spectral.metric == "spectral_centroid")].sort_values("layer")["mean"]) > 0))
    lines += ["", f"Layer means {'increase strictly' if increasing else 'do not increase strictly'} with depth. "
              f"Within-image last-minus-first centroid: {delta['mean']:.4f} "
              f"[{delta.ci025:.4f}, {delta.ci975:.4f}]. "
              f"{int(trends.strictly_increasing.sum())}/32 images have strictly increasing layer means. "
              "Interpret this as spatial-scale evidence, not evidence for particular facial semantics.", "",
              f"Zero-energy modes: {int(spectral[spectral.role == 'hidden'].zero_energy.sum())} hidden "
              f"and {int(spectral[spectral.role == 'output_rgb'].zero_energy.sum())} output cases. "
              "Final RGB players are reported separately in `spectral_output_rgb_summary.csv` and "
              "`figures/output_rgb_spatial_scale.*`; they are excluded from the hidden-layer trend.", "",
              "## Artifacts and limitations", "",
              "- `agreement_by_filter_image.csv`, `agreement_by_image_layer.csv`, and `agreement_by_layer.csv`: spatial results and paired gains.",
              "- `spectral_by_filter_image.csv`, `spectral_by_image_layer.csv`, `spectral_by_layer.csv`, and `spectral_trend_*.csv`: signed-map scale and image-level trends.",
              "- `figures/`: PNG/PDF comparisons, layer distributions, separate RGB results, and fixed-index examples (image 0, filter 0 in each hidden layer). Display panels use individual color ranges and support qualitative localization only.",
              "- `../cache/`: raw single-filter outputs, completion flags, baseline, copied latents, summed modes, and packed attribution masks. Input fingerprints guard against incompatible cache reuse.",
              "", "The 32 saved images are one fixed sample, and only one trained checkpoint is examined. "
              "No uncertainty over training seeds or historical Shapley permutations can be recovered "
              "from averaged modes. Frequency measurements depend on power weighting, image boundaries, "
              "and output resolution; the Hann check probes one boundary choice. Spatial masks fix the "
              "requested 20% threshold rather than selecting it based on the outcome. No manual semantic "
              "annotation or quantitative semantic claim is used.", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BASE / "saved_results/celebA_GAN_bigger_smodel.pth")
    parser.add_argument("--modes", type=Path, default=BASE / "saved_results/shapley_modes_bigger_celebA.pkl")
    parser.add_argument("--latents", type=Path, default=BASE / "fixed_noise.t")
    parser.add_argument("--output", type=Path, default=BASE / "quantification/results")
    parser.add_argument("--cache", type=Path, default=BASE / "quantification/cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2810)
    parser.add_argument("--fraction", type=float, default=.20)
    parser.add_argument("--shuffles", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--stage", choices=("all", "validate", "ablations", "modes", "agreement", "report"), default="all")
    args = parser.parse_args()
    if args.shuffles < 2 or args.bootstrap < 1 or args.batch_size < 1 or args.threads < 1 or args.fraction != .2:
        parser.error("Require shuffles >=2, bootstrap/threads/batch-size >=1, and fraction=0.20 for this analysis.")
    # Numerical work uses a bounded thread pool; report-only does not touch CUDA.
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        if args.stage == "report":
            identity = json.loads((args.cache / "identity.json").read_text())
            if any(identity["settings"][k] != getattr(args, k) for k in ("seed", "fraction", "shuffles", "bootstrap")):
                raise ValueError("Report settings must match the cached run.")
            data = summarize(args)
            figures(args, *data)
            report(args, data)
            log(f"Figures and report saved to {args.output}")
            return
        modes, reader = initialize(args)
        if args.stage in ("all", "validate", "ablations"):
            generator, latents = model_and_latents(args)
            validate(args, modes, generator, latents)
        elif not json.loads((args.output / "compatibility.json").read_text())["passed"]:
            raise ValueError("Validate compatibility before analysis")
        if args.stage in ("all", "ablations"):
            ablations(args, generator, latents)
        if args.stage in ("all", "modes"):
            mode_metrics(args, modes)
        if args.stage in ("all", "agreement"):
            agreement(args)
        if args.stage == "all":
            data = summarize(args)
            figures(args, *data)
            report(args, data)
            log(f"Completed: {args.output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
