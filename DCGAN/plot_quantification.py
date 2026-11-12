#!/usr/bin/env python3
"""Redraw saved DCGAN results in the shared MLP/LLM publication style.

No inference or new estimates are performed. Run before build_illustrated_report.py.
Exports editable vector PDF/SVG and 600-dpi PNG for each of the six figures.
"""

import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dcgan-quantification-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from quantification.storage import load_modes, sha256

BASE = Path(__file__).resolve().parent
ROOT = BASE / "quantification"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
FIGURES = RESULTS / "figures"
STYLE_PATH = BASE.parent / "MLP/visual_config.py"
spec = importlib.util.spec_from_file_location("msa_reference_style", STYLE_PATH)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def apply_style():
    style.set_visual_style()
    families = {entry.name for entry in font_manager.fontManager.ttflist}
    # The reference code requests Atkinson; the checked-in reference PDFs use
    # DejaVu Sans. Make that fallback explicit when Atkinson is unavailable.
    family = "Atkinson Hyperlegible" if "Atkinson Hyperlegible" in families else "DejaVu Sans"
    mpl.rcParams.update({"font.family": family, "font.sans-serif": [family],
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                         "savefig.dpi": 600, "legend.frameon": False})
    return family


def tidy(ax, ylabel, title=None, zero=True):
    ax.set(xlabel="Hidden layer", ylabel=ylabel, xticks=range(1, 6))
    if title:
        ax.set_title(title)
    if zero:
        ax.axhline(0, color=style.GRAY, lw=.6, ls="--", zorder=0)
    sns.despine(ax=ax, offset=1, trim=True)


def save(fig, name, layout=True):
    if layout:
        fig.tight_layout(pad=.6)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=600, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def intervals(ax, frame, color, label=None, offset=0, factor=1, marker="."):
    mean = frame["mean"].to_numpy() * factor
    error = np.maximum(0, np.vstack([frame["mean"] - frame.ci025, frame.ci975 - frame["mean"]])) * factor
    ax.errorbar(frame.layer.to_numpy() + 1 + offset, mean, yerr=error,
                fmt=marker, color=color, label=label, markersize=6, capsize=1,
                elinewidth=.8, markeredgewidth=.6, zorder=5)


def main():
    FIGURES.mkdir(exist_ok=True)
    family = apply_style()
    spatial = pd.read_csv(RESULTS / "agreement_by_layer.csv")
    images = pd.read_csv(RESULTS / "agreement_by_image_layer.csv")
    scale = pd.read_csv(RESULTS / "spectral_by_layer.csv")
    scale_images = pd.read_csv(RESULTS / "spectral_by_image_layer.csv")
    modes_table = pd.read_csv(RESULTS / "spectral_by_filter_image.csv")
    sensitivity = pd.read_csv(RESULTS / "agreement_sensitivity_by_layer.csv")
    consumed = ["agreement_by_layer.csv", "agreement_by_image_layer.csv", "spectral_by_layer.csv",
                "spectral_by_image_layer.csv", "spectral_by_filter_image.csv",
                "agreement_sensitivity_by_layer.csv", "spectral_output_rgb_summary.csv"]

    fig, ax = plt.subplots(figsize=(18 * style.CM, 8.5 * style.CM))
    for offset, metric, label, color in (
        (-.13, "capture", "Same filter", style.NIGHT_BLUE),
        (0, "mismatched_mean", "Other filters in layer", style.PURPLE),
        (.13, "shuffle_mc_mean", "Shuffled locations", style.GRAY),
    ):
        intervals(ax, spatial[spatial.metric == metric].sort_values("layer"), color, label, offset)
    ax.axhline(3277 / 16384, color=style.GRAY, lw=.6, ls="--", zorder=0)
    ax.set(ylim=(0, .8), yticks=np.arange(0, .81, .2))
    tidy(ax, "Image change captured (%)", "Top 20% of attribution pixels", zero=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0, symbol=""))
    ax.legend(loc="upper right")
    save(fig, "spatial_agreement")

    fig, axes = plt.subplots(1, 2, figsize=(18 * style.CM, 8.5 * style.CM), sharey=True)
    rng = np.random.default_rng(2810)
    for ax, metric, title, color in zip(axes, ("gain_vs_shuffle", "gain_vs_mismatched"),
                                       ("Compared with shuffled locations", "Compared with other filters"),
                                       (style.PURPLE, style.NIGHT_BLUE)):
        for layer, group in images.groupby("layer"):
            ax.scatter(layer + 1 + rng.uniform(-.13, .13, len(group)), group[metric] * 100,
                       s=6, color=color, alpha=.55, linewidths=0, zorder=2)
        intervals(ax, spatial[spatial.metric == metric].sort_values("layer"), style.HALF_BLACK,
                  factor=100, marker="_")
        ax.set(ylim=(-2, 65), yticks=(0, 20, 40, 60))
        tidy(ax, "Extra change captured (percentage points)" if ax is axes[0] else "", title)
    save(fig, "spatial_agreement_by_image")

    fig, axes = plt.subplots(1, 2, figsize=(18 * style.CM, 8.5 * style.CM), sharey=True)
    for ax, metric, title in zip(axes, ("gain_vs_shuffle", "gain_vs_mismatched"),
                                ("Compared with shuffled locations", "Compared with other filters")):
        for threshold, offset, color, label in (
            (0, -.13, style.NIGHT_BLUE, "All positive changes"),
            (1e-6, 0, style.PURPLE, r"Mean change $\geq 10^{-6}$"),
            (1e-5, .13, style.GRAY, r"Mean change $\geq 10^{-5}$"),
        ):
            subset = sensitivity[(sensitivity.metric == metric) & (sensitivity.minimum_mean_change == threshold)]
            intervals(ax, subset.sort_values("layer"), color, label, offset, factor=100)
        ax.set(ylim=(-2, 65), yticks=(0, 20, 40, 60))
        tidy(ax, "Extra change captured (percentage points)" if ax is axes[0] else "", title)
    axes[0].legend(loc="lower left", fontsize=7)
    save(fig, "spatial_numerical_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(18 * style.CM, 8.5 * style.CM), sharey=True)
    hidden = modes_table[modes_table.role == "hidden"]
    data = [hidden.loc[hidden.layer == layer, "spectral_centroid"].dropna().to_numpy() for layer in range(5)]
    violin = axes[0].violinplot(data, positions=range(1, 6), showmedians=True)
    for body in violin["bodies"]:
        body.set(facecolor=style.PURPLE, edgecolor="none", alpha=.45)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        violin[key].set(color=style.NIGHT_BLUE, linewidth=.7)
    for _, group in scale_images[scale_images.layer < 5].groupby("image"):
        axes[1].plot(group.layer + 1, group.spectral_centroid, color=style.GRAY, alpha=.45, lw=.45, zorder=1)
    for metric, offset, label, color in (("spectral_centroid", -.05, "Standard", style.NIGHT_BLUE),
                                        ("hann_centroid", .05, "Reduced edge effects", style.PURPLE)):
        group = scale[(scale.layer < 5) & (scale.metric == metric)].sort_values("layer")
        intervals(axes[1], group, color, label, offset)
    for ax, title in zip(axes, ("All filter-image maps", "Average within each image")):
        ax.set(ylim=(0, .8), yticks=(0, .2, .4, .6, .8))
        tidy(ax, "Spatial frequency (normalized)" if ax is axes[0] else "", title, zero=False)
    axes[1].legend(loc="upper left", fontsize=7)
    save(fig, "layer_spatial_scale")

    fig, ax = plt.subplots(figsize=(8.5 * style.CM, 7.5 * style.CM))
    rgb = modes_table[modes_table.role == "output_rgb"]
    rgb_summary = pd.read_csv(RESULTS / "spectral_output_rgb_summary.csv").set_index("channel")
    for x, channel, color in zip(range(3), "RGB", (style.PURPLE, style.NIGHT_BLUE, style.GRAY)):
        values = rgb.loc[rgb.channel == channel, "spectral_centroid"].to_numpy()
        ax.scatter(x + rng.uniform(-.13, .13, len(values)), values, color=color, s=6, alpha=.65, linewidth=0)
        row = rgb_summary.loc[channel]
        ax.errorbar(x, row["mean"], yerr=[[row["mean"] - row.ci025], [row.ci975 - row["mean"]]],
                    fmt="_", color=style.HALF_BLACK, markersize=10, capsize=1, lw=.8)
    ax.set(xticks=range(3), xticklabels=list("RGB"), ylim=(0, .08), yticks=(0, .02, .04, .06, .08),
           xlabel="Output channel", ylabel="Spatial frequency (normalized)", title="Final RGB channels")
    sns.despine(ax=ax, offset=1, trim=True)
    save(fig, "output_rgb_spatial_scale")

    qualitative_examples()
    metadata = {"script_sha256": sha256(__file__), "reference_style": str(STYLE_PATH),
                "reference_style_sha256": sha256(STYLE_PATH), "font": family,
                "font_note": "Explicit fallback to the DejaVu Sans font embedded in the checked-in MLP/LLM reference PDFs when Atkinson is unavailable.",
                "font_size_pt": 8, "pdf_fonttype": 42, "svg_fonttype": "none", "png_dpi": 600,
                "display_layers": "1-5 correspond to CSV/checkpoint layers 0-4",
                "source_csv_sha256": {name: sha256(RESULTS / name) for name in consumed},
                "statistics_recomputed": False,
                "exports": sorted(p.name for p in FIGURES.iterdir() if p.suffix in (".png", ".svg", ".pdf"))}
    (RESULTS / "figure_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Six publication figures saved as PDF, SVG, and 600-dpi PNG.")


def qualitative_examples():
    image = 30
    # All IDs are from the author's pre-existing example in CelebA GAN.ipynb,
    # cells 19 and 25. They are illustrative choices, not a representative sample.
    units = [72, 50, 7, 12, 8]
    baseline = np.load(CACHE / "baseline.npy")
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    modes, reader = load_modes(manifest["inputs"]["modes"]["path"])
    fig, axes = plt.subplots(5, 5, figsize=(18 * style.CM, 18.5 * style.CM))
    fig.subplots_adjust(left=.12, right=.99, bottom=.10, top=.94, wspace=.045, hspace=.13)
    attr_cmap = sns.blend_palette([style.WHITE, style.PURPLE, style.PURPLER], as_cmap=True)
    change_cmap = sns.blend_palette([style.WHITE, style.BLUE, style.NIGHT_BLUE], as_cmap=True)
    records = []
    for layer, unit in enumerate(units):
        output = np.load(CACHE / f"ablations_layer_{layer}.npy", mmap_mode="r")[unit, image]
        attribution = np.abs(modes.get_shapley_mode((layer, unit))[image]).mean(axis=0)
        change = np.abs(baseline[image].astype(np.float64) - output).mean(axis=0)
        packed = np.load(CACHE / f"masks_layer_{layer}.npy", mmap_mode="r")[unit, image]
        mask = np.unpackbits(packed).reshape(128, 128).astype(bool)
        axes[layer, 0].imshow(baseline[image].transpose(1, 2, 0), interpolation="none")
        axes[layer, 1].imshow(output.transpose(1, 2, 0), interpolation="none")
        axes[layer, 2].imshow(attribution / attribution.max(), cmap=attr_cmap, vmin=0, vmax=1, interpolation="none")
        axes[layer, 3].imshow(change / change.max(), cmap=change_cmap, vmin=0, vmax=1, interpolation="none")
        axes[layer, 4].imshow(baseline[image].transpose(1, 2, 0), interpolation="none")
        overlay = np.zeros((128, 128, 4))
        overlay[mask] = mpl.colors.to_rgba(style.PURPLE, .65)
        axes[layer, 4].imshow(overlay, interpolation="none")
        axes[layer, 0].set_ylabel(f"Layer {layer + 1}\nFilter {unit}", rotation=0, ha="right", va="center", labelpad=5)
        for ax in axes[layer]:
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
        records.append({"image": image, "csv_layer": layer, "display_layer": layer + 1, "filter": unit,
                        "attribution_display_max": float(attribution.max()), "change_display_max": float(change.max()),
                        "top_pixels": int(mask.sum())})
    for ax, label in zip(axes[0], ("Original", "Filter removed", "Attribution", "Observed change", "Selected 20%")):
        ax.set_title(label, pad=5)
    for x, cmap, label in ((.475, attr_cmap, "Attribution (relative)"), (.66, change_cmap, "Change (relative)")):
        cax = fig.add_axes([x, .047, .14, .012])
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmap), cax=cax, orientation="horizontal", ticks=(0, 1))
        bar.set_label(label, fontsize=7, labelpad=0)
        bar.ax.tick_params(labelsize=7, pad=1)
        bar.outline.set_visible(False)
    save(fig, "fixed_spatial_examples", layout=False)
    selection = {"selection": "The original notebook uses image 30 and lists three example filters per layer. One from each layer was chosen for visible spatial structure; no numerical agreement threshold was used.",
                 "interpretation": "Illustrative examples only. Population results use all images and filters.",
                 "display": "Each heatmap is divided by its own maximum only for display. Scientific calculations use raw unnormalized maps. Purple tint marks the actual 3277 selected pixels; no smoothing or threshold changes.",
                 "examples": records}
    (RESULTS / "qualitative_examples.json").write_text(json.dumps(selection, indent=2) + "\n")


if __name__ == "__main__":
    main()
