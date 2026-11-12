#!/usr/bin/env python3
"""Check whether numerically tiny single-filter changes affect spatial conclusions.

Run after quantify.py. Does not regenerate modes or interventions and needs no GPU.
Thresholds are in raw [0,1] output units, averaged over channels and pixels.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dcgan-quantification-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from quantification.metrics import bootstrap_mean
from quantification.storage import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "quantification/results")
    args = parser.parse_args()
    source = args.output / "agreement_by_filter_image.csv"
    frame = pd.read_csv(source)
    manifest = json.loads((args.output / "manifest.json").read_text())
    qc = json.loads((args.output / "compatibility.json").read_text())
    seed = manifest["settings"]["seed"]
    bootstrap = manifest["settings"]["bootstrap"]
    metrics = ("capture", "gain_vs_shuffle", "gain_vs_mismatched")
    image_rows = []
    rows = []
    for scenario, threshold in enumerate((0., 1e-6, 1e-5)):
        for (image, layer), group in frame.groupby(["image", "layer"]):
            valid = group[group.valid_agreement & (group.change_abs_mean >= threshold)]
            image_rows.append({"minimum_mean_change": threshold, "image": image, "layer": layer,
                               "retained_filters": len(valid), "all_filters": len(group),
                               **{metric: valid[metric].mean() for metric in metrics}})
        by_image = pd.DataFrame(image_rows)
        by_image = by_image[by_image.minimum_mean_change == threshold]
        for layer, group in by_image.groupby("layer"):
            for j, metric in enumerate(metrics):
                rng = np.random.default_rng(np.random.SeedSequence([seed, 7, scenario, int(layer), j]))
                mean, lo, hi = bootstrap_mean(group[metric], bootstrap, rng)
                rows.append({"minimum_mean_change": threshold, "layer": layer, "metric": metric,
                             "mean": mean, "ci025": lo, "ci975": hi,
                             "images": int(group[metric].notna().sum()),
                             "retained_cases": int(group.retained_filters.sum()),
                             "all_cases": int(group.all_filters.sum())})
    pd.DataFrame(image_rows).to_csv(args.output / "agreement_sensitivity_by_image_layer.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output / "agreement_sensitivity_by_layer.csv", index=False)
    provenance = {"agreement_csv_sha256": sha256(source), "script_sha256": sha256(__file__),
                  "minimum_mean_change_thresholds": [0., 1e-6, 1e-5],
                  "seed": seed, "bootstrap": bootstrap,
                  "reason": "Historical cuDNN kernels exhibit sub-micro-unit repeat differences; tiny changes can be dominated by numerical noise."}
    (args.output / "sensitivity_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(axes, metrics[1:], ("Gain over spatial shuffle", "Gain over other filters")):
        for threshold, marker, label in ((0., "o", "All positive changes"),
                                           (1e-6, "s", "Mean change ≥ 10⁻⁶"),
                                           (1e-5, "^", "Mean change ≥ 10⁻⁵")):
            group = summary[(summary.minimum_mean_change == threshold) & (summary.metric == metric)].sort_values("layer")
            ax.errorbar(group.layer, group["mean"],
                        yerr=np.maximum(0, np.vstack([group["mean"] - group.ci025, group.ci975 - group["mean"]])),
                        fmt=marker + "-", capsize=3, label=label)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set(xticks=range(5), xlabel="Hidden layer", ylabel="Capture difference", title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.output / "figures" / f"spatial_numerical_sensitivity.{extension}", dpi=180)
    plt.close(fig)
    lines = ["# Spatial agreement: numerical sensitivity", "",
             f"Repeated unlesioned inference differed by at most {qc['max_repeat_difference']:.3g} "
             "in an individual raw output value. Exact zero-change detection can therefore miss "
             "functionally unchanged filters when cuDNN kernels differ at floating-point precision. "
             "The primary analysis retains all positive changes as specified. This supplementary "
             "analysis excludes targets with mean absolute image change below 10⁻⁶ or 10⁻⁵, "
             "in raw [0,1] output units. These thresholds are conservative numerical checks; "
             "they do not establish a biological or perceptual relevance threshold.", "",
             "Each scenario averages retained filters within images and bootstraps images with "
             f"{bootstrap:,} resamples. The original spatial masks and null controls are unchanged; "
             "a small-change filter's attribution map can still serve as a same-layer mismatch "
             "reference. Intervals are paired, conditional on the saved estimates and checkpoint.", "",
             "| Minimum mean change | Layer | Retained/total cases | Gain vs shuffle (95% CI) | Gain vs other filters (95% CI) |",
             "|---:|---:|---:|---:|---:|"]
    for threshold in (0., 1e-6, 1e-5):
        for layer in range(5):
            group = summary[(summary.minimum_mean_change == threshold) & (summary.layer == layer)]
            a = group[group.metric == "gain_vs_shuffle"].iloc[0]
            b = group[group.metric == "gain_vs_mismatched"].iloc[0]
            lines.append(f"| {threshold:g} | {layer} | {a.retained_cases:.0f}/{a.all_cases:.0f} | "
                         f"{a['mean']:.4f} [{a.ci025:.4f}, {a.ci975:.4f}] | "
                         f"{b['mean']:.4f} [{b.ci025:.4f}, {b.ci975:.4f}] |")
    lines += ["", "Positive lower interval bounds indicate image-level mean agreement above the "
              "control under that inclusion rule. Compare rows within each layer to assess "
              "whether primary conclusions depend on numerically tiny effects. "
              "This sensitivity check does not alter the primary dataset or regenerate Shapley Modes.", ""]
    (args.output / "SENSITIVITY.md").write_text("\n".join(lines))
    print(args.output / "SENSITIVITY.md")


if __name__ == "__main__":
    main()
