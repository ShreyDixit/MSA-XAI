from __future__ import annotations

import argparse
import logging
import math
import re
from glob import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from visual_config import BLUE, CM, HALF_BLACK, NIGHT_BLUE, ORANGE, PURPLE, set_visual_style

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_GLOB = "*_save_permutations_true_num_permutations_50_*_permutation_contributions.csv"
DEFAULT_WINDOW_SIZE = 50
REQUIRED_COLUMNS = {
    "Permutation",
    "Metric_Index",
    "Contribution",
    "Performance Metric",
    "Layer",
    "Expert",
}
ELEMENT_COLUMNS = ["Performance Metric", "Layer", "Expert"]
DOMAIN_BY_METRIC = {
    "addition_accuracy": "Arithmetic",
    "subtraction_accuracy": "Arithmetic",
    "multiplication_accuracy": "Arithmetic",
    "english_accuracy": "Language",
    "german_accuracy": "Language",
    "french_accuracy": "Language",
    "spanish_accuracy": "Language",
    "italian_accuracy": "Language",
    "country_capital_accuracy": "Factual Knowledge",
}
DOMAIN_ORDER = ["Arithmetic", "Language", "Factual Knowledge"]
DOMAIN_PALETTE = {
    "Arithmetic": NIGHT_BLUE,
    "Language": PURPLE,
    "Factual Knowledge": ORANGE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Mixtral per-permutation Shapley CSVs into one long run and "
            "plot combined-permutation convergence."
        )
    )
    parser.add_argument(
        "--input-glob",
        default=str(SCRIPT_DIR / DEFAULT_INPUT_GLOB),
        help=(
            "Glob for *_permutation_contributions.csv files. Defaults to the "
            "50-permutation Mixtral files in LLM/. Quote this argument if it "
            "contains wildcards."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "Figures" / "convergence",
        help="Directory for generated figures and combined convergence CSVs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional subset of Performance Metric names to include before domain averaging.",
    )
    parser.add_argument(
        "--no-domain-average",
        action="store_true",
        help="Plot individual metrics instead of averaging them into domains.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help=(
            "Number of largest absolute final-contribution experts to highlight "
            "in the optional trajectory plot."
        ),
    )
    parser.add_argument(
        "--rank-top-k",
        nargs="+",
        type=int,
        default=[10, 20, 50],
        help="Top-k expert set sizes for rank-stability analyses.",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[100, 250, 500, 750, 1000],
        help="Permutation counts for sensitivity summaries.",
    )
    parser.add_argument(
        "--top-experts",
        type=int,
        default=15,
        help="Number of top absolute-contribution experts to show in CI plots per domain.",
    )
    parser.add_argument(
        "--ci-reference-lines",
        nargs="+",
        type=float,
        default=[0.005, 0.01],
        help="Horizontal reference lines for final CI-width distribution plots.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output figure formats. Defaults to pdf and png.",
    )
    parser.add_argument(
        "--plot-trajectories",
        action="store_true",
        help="Also plot expert running-Shapley trajectories for the combined run.",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=None,
        help=(
            "Optional fixed absolute tolerance on the metric scale. If omitted, "
            "the script computes a metric-specific tolerance as the sum of the "
            "final Shapley values across experts divided by the number of experts."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help=(
            "Window size for the optional stability diagnostic: compare the current "
            "running estimate with the estimate this many permutations earlier."
        ),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of input files, useful for quick checks.",
    )
    return parser.parse_args()


def run_label_from_path(path: Path) -> str:
    timestamp_match = re.search(r"(\d{8}-\d{6})", path.stem)
    if timestamp_match:
        return timestamp_match.group(1)

    trial_match = re.search(r"trial[_-]?(\d+)", path.stem)
    if trial_match:
        return f"trial_{trial_match.group(1)}"

    return path.stem.replace("_permutation_contributions", "")


def discover_files(input_glob: str, max_files: int | None) -> list[Path]:
    paths = sorted(Path(path) for path in glob(input_glob))
    paths = [path for path in paths if path.is_file()]
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise FileNotFoundError(f"No files matched --input-glob {input_glob!r}")
    return paths


def load_run(path: Path, run_order: int, metrics: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    if metrics is not None:
        df = df[df["Performance Metric"].isin(metrics)].copy()
        if df.empty:
            raise ValueError(f"{path} has none of the requested metrics: {metrics}")

    df["Run"] = run_label_from_path(path)
    df["Run Order"] = run_order
    df["Source File"] = path.name
    df["Permutation"] = df["Permutation"].astype(int)
    df["Layer"] = df["Layer"].astype(int)
    df["Expert"] = df["Expert"].astype(int)
    df["Contribution"] = df["Contribution"].astype(float)
    return df


def average_domains(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Domain"] = df["Performance Metric"].map(DOMAIN_BY_METRIC)
    missing_metrics = sorted(df.loc[df["Domain"].isna(), "Performance Metric"].unique())
    if missing_metrics:
        raise ValueError(
            "No domain mapping defined for metrics: " + ", ".join(missing_metrics)
        )

    averaged = (
        df.groupby(
            [
                "Run",
                "Run Order",
                "Source File",
                "Permutation",
                "Layer",
                "Expert",
                "Domain",
            ],
            as_index=False,
        )["Contribution"]
        .mean()
        .rename(columns={"Domain": "Performance Metric"})
    )
    return averaged


def load_combined(
    paths: list[Path],
    metrics: list[str] | None,
    window_size: int,
    domain_average: bool,
) -> pd.DataFrame:
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    runs = [load_run(path, run_order, metrics) for run_order, path in enumerate(paths)]
    combined = pd.concat(runs, ignore_index=True)
    if domain_average:
        combined = average_domains(combined)
    combined.sort_values(
        ELEMENT_COLUMNS + ["Run Order", "Permutation"],
        inplace=True,
    )

    combined["Num Permutations"] = combined.groupby(ELEMENT_COLUMNS).cumcount() + 1
    grouped_contributions = combined.groupby(ELEMENT_COLUMNS)["Contribution"]
    combined["Running Contribution"] = grouped_contributions.transform(
        lambda values: values.expanding().mean()
    )
    combined["Running Std"] = grouped_contributions.transform(
        lambda values: values.expanding().std(ddof=1)
    )
    combined["Running SE"] = combined["Running Std"] / combined[
        "Num Permutations"
    ].pow(0.5)
    combined["Running CI95 Half Width"] = 1.96 * combined["Running SE"]
    combined["Window Previous Estimate"] = combined.groupby(ELEMENT_COLUMNS)[
        "Running Contribution"
    ].shift(window_size)
    combined["Window Abs Change"] = (
        combined["Running Contribution"] - combined["Window Previous Estimate"]
    ).abs()

    final = combined.groupby(ELEMENT_COLUMNS, as_index=False).tail(1)[
        ELEMENT_COLUMNS + ["Running Contribution"]
    ]
    final = final.rename(columns={"Running Contribution": "Final Contribution"})

    combined = combined.merge(final, on=ELEMENT_COLUMNS, how="left")
    combined["Delta To Final"] = (
        combined["Running Contribution"] - combined["Final Contribution"]
    )
    combined["Abs Delta To Final"] = combined["Delta To Final"].abs()
    combined["Element"] = "L" + combined["Layer"].astype(str) + " E" + combined["Expert"].astype(str)
    return combined


def compute_metric_tolerances(
    combined: pd.DataFrame,
    absolute_tolerance: float | None,
) -> pd.DataFrame:
    final_estimates = combined.groupby(ELEMENT_COLUMNS, as_index=False).tail(1)
    metric_summary = (
        final_estimates.groupby("Performance Metric")
        .agg(
            TotalFinalShapley=("Final Contribution", "sum"),
            NumExperts=("Final Contribution", "size"),
        )
        .reset_index()
    )
    metric_summary["MeanFinalShapley"] = (
        metric_summary["TotalFinalShapley"] / metric_summary["NumExperts"]
    )
    metric_summary["MeanAbsFinalShapley"] = (
        final_estimates.assign(AbsFinalShapley=final_estimates["Final Contribution"].abs())
        .groupby("Performance Metric")["AbsFinalShapley"]
        .mean()
        .reindex(metric_summary["Performance Metric"])
        .to_numpy()
    )

    if absolute_tolerance is None:
        metric_summary["AbsoluteTolerance"] = metric_summary["MeanFinalShapley"].abs()
        metric_summary["ToleranceSource"] = "abs(sum_final_shapley / num_experts)"
    else:
        metric_summary["AbsoluteTolerance"] = absolute_tolerance
        metric_summary["ToleranceSource"] = "fixed_cli_value"

    return metric_summary


def make_diagnostics(
    combined: pd.DataFrame,
    metric_tolerances: pd.DataFrame,
) -> pd.DataFrame:
    combined = combined.merge(
        metric_tolerances[["Performance Metric", "AbsoluteTolerance", "ToleranceSource"]],
        on="Performance Metric",
        how="left",
    )
    combined["BelowAbsoluteTolerance"] = combined["Running CI95 Half Width"].le(
        combined["AbsoluteTolerance"]
    )

    return (
        combined.groupby(["Performance Metric", "Num Permutations"])
        .agg(
            MeanAbsDeltaToFinal=("Abs Delta To Final", "mean"),
            MeanAbsWindowChange=("Window Abs Change", "mean"),
            MeanSE=("Running SE", "mean"),
            MeanCI95HalfWidth=("Running CI95 Half Width", "mean"),
            MedianCI95HalfWidth=("Running CI95 Half Width", "median"),
            P90CI95HalfWidth=("Running CI95 Half Width", lambda values: values.quantile(0.90)),
            P95CI95HalfWidth=("Running CI95 Half Width", lambda values: values.quantile(0.95)),
            MaxCI95HalfWidth=("Running CI95 Half Width", "max"),
            FractionBelowAbsoluteTolerance=("BelowAbsoluteTolerance", "mean"),
            AbsoluteTolerance=("AbsoluteTolerance", "first"),
            ToleranceSource=("ToleranceSource", "first"),
        )
        .reset_index()
    )


def facet_grid(metrics: list[str]) -> tuple[plt.Figure, list[plt.Axes]]:
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(18 * CM, max(8 * CM, 3.8 * len(metrics) * CM)),
        dpi=300,
        sharex=True,
        squeeze=False,
    )
    return fig, list(axes.ravel())


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def save_figure(fig: plt.Figure, output_base: Path, formats: list[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_base.with_suffix(f".{extension}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_metric_convergence(
    axis: plt.Axes,
    metric_df: pd.DataFrame,
    metric: str,
) -> None:
    absolute_tolerance = float(metric_df["AbsoluteTolerance"].iloc[0])
    axis.plot(
        metric_df["Num Permutations"],
        metric_df["FractionBelowAbsoluteTolerance"],
        color=NIGHT_BLUE,
        label=f"CI half-width <= {absolute_tolerance:g}",
    )
    axis.axhline(0.95, color=ORANGE, linewidth=0.8, linestyle="--", label="95% of experts")
    axis.set_title(metric, loc="left")
    axis.set_ylabel("Fraction of experts")
    axis.set_ylim(-0.02, 1.02)
    axis.margins(x=0.01)
    sns.despine(ax=axis, trim=True)


def ordered_labels(labels: list[str]) -> list[str]:
    order = {label: index for index, label in enumerate(DOMAIN_ORDER)}
    return sorted(labels, key=lambda label: order.get(label, len(order)))


def plot_domain_convergence(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
) -> None:
    labels = ordered_labels(list(diagnostics["Performance Metric"].drop_duplicates()))
    max_permutations = int(diagnostics["Num Permutations"].max())

    fig, axis = plt.subplots(figsize=(12 * CM, 8 * CM), dpi=300)
    for label in labels:
        label_df = diagnostics[diagnostics["Performance Metric"] == label]
        tolerance = float(label_df["AbsoluteTolerance"].iloc[0])
        axis.plot(
            label_df["Num Permutations"],
            label_df["FractionBelowAbsoluteTolerance"],
            color=DOMAIN_PALETTE.get(label, BLUE),
            linewidth=1.5,
            label=f"{label} (tol. {tolerance:.3g})",
        )

    axis.axhline(0.95, color=HALF_BLACK, linewidth=0.8, linestyle="--", label="95% threshold")
    axis.set_xlabel("Permutations")
    axis.set_ylabel("Experts below domain tolerance")
    axis.set_ylim(-0.02, 1.02)
    axis.margins(x=0.01)
    axis.legend(frameon=False, loc="lower right")
    sns.despine(ax=axis, trim=True)
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / f"combined_{max_permutations}_domain_stability", formats)


def plot_combined_convergence(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    domain_average: bool,
) -> None:
    if domain_average:
        plot_domain_convergence(diagnostics, output_dir, formats)
        return

    metrics = list(diagnostics["Performance Metric"].drop_duplicates())
    fig, axes = facet_grid(metrics)

    for axis, metric in zip(axes, metrics):
        metric_df = diagnostics[diagnostics["Performance Metric"] == metric]
        plot_metric_convergence(axis, metric_df, metric)

    axes[-1].set_xlabel("Permutations")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    max_permutations = int(diagnostics["Num Permutations"].max())
    fig.suptitle(
        f"Combined {max_permutations}-Permutation Precision: metric-specific average contribution tolerance",
        y=0.965,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.1)
    save_figure(fig, output_dir / f"combined_{max_permutations}_stability", formats)

    metric_dir = output_dir / "per_metric"
    for metric in metrics:
        metric_df = diagnostics[diagnostics["Performance Metric"] == metric]
        metric_fig, metric_axis = plt.subplots(figsize=(12 * CM, 7 * CM), dpi=300)
        plot_metric_convergence(metric_axis, metric_df, metric)
        metric_axis.set_xlabel("Permutations")
        metric_axis.legend(frameon=False, ncol=1)
        metric_fig.tight_layout(pad=0.2)
        save_figure(metric_fig, metric_dir / f"combined_stability_{safe_filename(metric)}", formats)


def plot_combined_trajectories(
    combined: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    top_k: int,
) -> None:
    metrics = list(combined["Performance Metric"].drop_duplicates())
    fig, axes = facet_grid(metrics)

    for axis, metric in zip(axes, metrics):
        metric_df = combined[combined["Performance Metric"] == metric]
        top_elements = (
            metric_df.groupby(["Layer", "Expert"], as_index=False)["Final Contribution"]
            .last()
            .assign(abs_final=lambda values: values["Final Contribution"].abs())
            .nlargest(top_k, "abs_final")[["Layer", "Expert"]]
        )
        top_keys = set(map(tuple, top_elements[["Layer", "Expert"]].to_numpy()))

        for (_, element_df) in metric_df.groupby(["Layer", "Expert"], sort=False):
            axis.plot(
                element_df["Num Permutations"],
                element_df["Running Contribution"],
                color="#B8B8B8",
                linewidth=0.35,
                alpha=0.18,
            )

        palette = sns.color_palette("tab10", n_colors=max(1, min(top_k, 10)))
        for color_idx, ((layer, expert), element_df) in enumerate(
            metric_df.groupby(["Layer", "Expert"], sort=False)
        ):
            if (layer, expert) not in top_keys:
                continue
            axis.plot(
                element_df["Num Permutations"],
                element_df["Running Contribution"],
                linewidth=1.0,
                alpha=0.95,
                color=palette[color_idx % len(palette)],
            )

        axis.set_title(metric)
        axis.set_xlabel("Permutations")
        axis.set_ylabel("Running Shapley estimate")
        axis.axhline(0, color=HALF_BLACK, linewidth=0.5, alpha=0.4)
        sns.despine(ax=axis, trim=True)

    axes[-1].set_xlabel("Permutations")
    fig.suptitle("Combined Expert Trajectories", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.1)
    save_figure(fig, output_dir / "combined_expert_trajectories", formats)


def final_expert_estimates(combined: pd.DataFrame) -> pd.DataFrame:
    return combined.groupby(ELEMENT_COLUMNS, as_index=False).tail(1).copy()


def plot_final_ci_width_distribution(
    combined: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    ci_reference_lines: list[float],
) -> None:
    final_df = final_expert_estimates(combined)
    labels = ordered_labels(list(final_df["Performance Metric"].drop_duplicates()))
    max_permutations = int(final_df["Num Permutations"].max())

    fig, axis = plt.subplots(figsize=(12 * CM, 8 * CM), dpi=300)
    sns.boxplot(
        data=final_df,
        x="Performance Metric",
        y="Running CI95 Half Width",
        hue="Performance Metric",
        order=labels,
        hue_order=labels,
        palette={label: DOMAIN_PALETTE.get(label, BLUE) for label in labels},
        width=0.55,
        fliersize=1.5,
        linewidth=0.8,
        legend=False,
        ax=axis,
    )
    sns.stripplot(
        data=final_df,
        x="Performance Metric",
        y="Running CI95 Half Width",
        order=labels,
        color=HALF_BLACK,
        alpha=0.22,
        size=1.6,
        jitter=0.22,
        ax=axis,
    )
    for reference in ci_reference_lines:
        axis.axhline(
            reference,
            color=HALF_BLACK,
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
        )
        axis.text(
            len(labels) - 0.52,
            reference,
            f" {reference:g}",
            va="bottom",
            ha="left",
            fontsize=7,
            color=HALF_BLACK,
        )

    axis.set_xlabel("")
    axis.set_ylabel("95% MC CI half-width")
    axis.set_title(f"Final Monte Carlo uncertainty at {max_permutations} permutations", loc="left")
    sns.despine(ax=axis, trim=True)
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / f"combined_{max_permutations}_final_ci_width_distribution", formats)


def plot_window_stability(
    diagnostics: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    window_size: int,
) -> None:
    labels = ordered_labels(list(diagnostics["Performance Metric"].drop_duplicates()))
    max_permutations = int(diagnostics["Num Permutations"].max())

    fig, axis = plt.subplots(figsize=(12 * CM, 8 * CM), dpi=300)
    for label in labels:
        label_df = diagnostics[diagnostics["Performance Metric"] == label].dropna(
            subset=["MeanAbsWindowChange"]
        )
        axis.plot(
            label_df["Num Permutations"],
            label_df["MeanAbsWindowChange"],
            color=DOMAIN_PALETTE.get(label, BLUE),
            linewidth=1.5,
            label=label,
        )

    axis.set_xlabel("Permutations")
    axis.set_ylabel(f"Mean |SV(n) - SV(n-{window_size})|")
    axis.set_title("Running estimate stability", loc="left")
    axis.legend(frameon=False)
    axis.margins(x=0.01)
    sns.despine(ax=axis, trim=True)
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / f"combined_{max_permutations}_window_stability", formats)


def rank_stability_table(combined: pd.DataFrame, top_ks: list[int]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    final_df = final_expert_estimates(combined)

    for label in ordered_labels(list(combined["Performance Metric"].drop_duplicates())):
        final_label = final_df[final_df["Performance Metric"] == label].copy()
        final_label["ExpertKey"] = list(zip(final_label["Layer"], final_label["Expert"]))
        final_scores = final_label.set_index("ExpertKey")["Final Contribution"]
        final_abs = final_scores.abs().sort_values(ascending=False)
        final_top = {top_k: set(final_abs.head(top_k).index) for top_k in top_ks}

        label_df = combined[combined["Performance Metric"] == label].copy()
        label_df["ExpertKey"] = list(zip(label_df["Layer"], label_df["Expert"]))
        for n_permutations, n_df in label_df.groupby("Num Permutations", sort=True):
            current_scores = n_df.set_index("ExpertKey")["Running Contribution"].reindex(
                final_scores.index
            )
            spearman = current_scores.rank().corr(final_scores.rank())
            record: dict[str, object] = {
                "Performance Metric": label,
                "Num Permutations": int(n_permutations),
                "SpearmanToFinal": spearman,
            }
            current_abs = current_scores.abs().sort_values(ascending=False)
            for top_k in top_ks:
                current_top = set(current_abs.head(top_k).index)
                record[f"Top{top_k}Overlap"] = len(current_top & final_top[top_k]) / top_k
            records.append(record)

    return pd.DataFrame.from_records(records)


def plot_rank_stability(
    rank_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    top_ks: list[int],
) -> None:
    labels = ordered_labels(list(rank_df["Performance Metric"].drop_duplicates()))
    max_permutations = int(rank_df["Num Permutations"].max())
    primary_top_k = top_ks[min(1, len(top_ks) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 7 * CM), dpi=300, sharex=True)
    for label in labels:
        label_df = rank_df[rank_df["Performance Metric"] == label]
        color = DOMAIN_PALETTE.get(label, BLUE)
        axes[0].plot(
            label_df["Num Permutations"],
            label_df["SpearmanToFinal"],
            color=color,
            linewidth=1.5,
            label=label,
        )
        axes[1].plot(
            label_df["Num Permutations"],
            label_df[f"Top{primary_top_k}Overlap"],
            color=color,
            linewidth=1.5,
            label=label,
        )

    axes[0].set_ylabel("Spearman vs final ranking")
    axes[1].set_ylabel(f"Top-{primary_top_k} overlap vs final")
    for axis in axes:
        axis.set_xlabel("Permutations")
        axis.set_ylim(-0.02, 1.02)
        axis.margins(x=0.01)
        sns.despine(ax=axis, trim=True)
    axes[0].set_title("Rank correlation", loc="left")
    axes[1].set_title("Top-k set stability", loc="left")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / f"combined_{max_permutations}_rank_stability", formats)


def checkpoint_sensitivity_table(
    rank_df: pd.DataFrame,
    checkpoints: list[int],
    top_ks: list[int],
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    available = sorted(rank_df["Num Permutations"].unique())
    for checkpoint in sorted(set(checkpoints)):
        eligible = [value for value in available if value <= checkpoint]
        if not eligible:
            continue
        actual = max(eligible)
        checkpoint_df = rank_df[rank_df["Num Permutations"] == actual].copy()
        checkpoint_df["Requested Checkpoint"] = checkpoint
        checkpoint_df["Actual Checkpoint"] = actual
        records.append(checkpoint_df)

    if not records:
        columns = ["Performance Metric", "Num Permutations", "SpearmanToFinal"] + [
            f"Top{top_k}Overlap" for top_k in top_ks
        ]
        return pd.DataFrame(columns=columns)
    return pd.concat(records, ignore_index=True)


def plot_checkpoint_sensitivity(
    checkpoint_df: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    top_ks: list[int],
) -> None:
    if checkpoint_df.empty:
        return

    labels = ordered_labels(list(checkpoint_df["Performance Metric"].drop_duplicates()))
    max_permutations = int(checkpoint_df["Actual Checkpoint"].max())
    primary_top_k = top_ks[min(1, len(top_ks) - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(18 * CM, 7 * CM), dpi=300, sharex=True)
    for label in labels:
        label_df = checkpoint_df[checkpoint_df["Performance Metric"] == label]
        color = DOMAIN_PALETTE.get(label, BLUE)
        axes[0].plot(
            label_df["Actual Checkpoint"],
            label_df["SpearmanToFinal"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            label=label,
        )
        axes[1].plot(
            label_df["Actual Checkpoint"],
            label_df[f"Top{primary_top_k}Overlap"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            label=label,
        )

    axes[0].set_ylabel("Spearman vs final ranking")
    axes[1].set_ylabel(f"Top-{primary_top_k} overlap vs final")
    for axis in axes:
        axis.set_xlabel("Permutations")
        axis.set_ylim(-0.02, 1.02)
        sns.despine(ax=axis, trim=True)
    axes[0].set_title("Checkpoint rank sensitivity", loc="left")
    axes[1].set_title("Checkpoint top-k sensitivity", loc="left")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.2)
    save_figure(fig, output_dir / f"combined_{max_permutations}_checkpoint_sensitivity", formats)


def plot_top_expert_cis(
    combined: pd.DataFrame,
    output_dir: Path,
    formats: list[str],
    top_experts: int,
) -> None:
    final_df = final_expert_estimates(combined)
    labels = ordered_labels(list(final_df["Performance Metric"].drop_duplicates()))
    max_permutations = int(final_df["Num Permutations"].max())

    fig, axes = plt.subplots(
        len(labels),
        1,
        figsize=(14 * CM, max(8 * CM, 5.0 * len(labels) * CM)),
        dpi=300,
        squeeze=False,
    )
    axes_list = list(axes.ravel())
    for axis, label in zip(axes_list, labels):
        label_df = final_df[final_df["Performance Metric"] == label].copy()
        label_df["AbsFinalContribution"] = label_df["Final Contribution"].abs()
        label_df = label_df.nlargest(top_experts, "AbsFinalContribution").sort_values(
            "Final Contribution"
        )
        y_positions = range(len(label_df))
        y_labels = [f"L{layer} E{expert}" for layer, expert in zip(label_df["Layer"], label_df["Expert"])]
        axis.errorbar(
            label_df["Final Contribution"],
            y_positions,
            xerr=label_df["Running CI95 Half Width"],
            fmt="o",
            markersize=3,
            linewidth=0.8,
            capsize=2,
            color=DOMAIN_PALETTE.get(label, BLUE),
            ecolor=HALF_BLACK,
        )
        axis.axvline(0, color=HALF_BLACK, linewidth=0.6, alpha=0.6)
        axis.set_yticks(list(y_positions), y_labels)
        axis.set_title(label, loc="left")
        axis.set_xlabel("Final Shapley estimate with 95% MC CI")
        sns.despine(ax=axis, trim=True, left=False)

    fig.tight_layout(pad=0.2, h_pad=1.0)
    save_figure(fig, output_dir / f"combined_{max_permutations}_top_expert_cis", formats)


def main() -> None:
    args = parse_args()
    set_visual_style()

    input_paths = discover_files(args.input_glob, args.max_files)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Combining {len(input_paths)} permutation contribution files:")
    for path in input_paths:
        print(f"  {path}")

    domain_average = not args.no_domain_average
    combined = load_combined(input_paths, args.metrics, args.window_size, domain_average)
    max_permutations = int(combined["Num Permutations"].max())
    metric_tolerances = compute_metric_tolerances(combined, args.absolute_tolerance)
    diagnostics = make_diagnostics(combined, metric_tolerances)

    combined_path = args.output_dir / f"combined_{max_permutations}_running_contributions.csv"
    diagnostics_path = args.output_dir / f"combined_{max_permutations}_stability_diagnostics.csv"
    tolerance_path = args.output_dir / f"combined_{max_permutations}_metric_tolerances.csv"
    combined.to_csv(combined_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    metric_tolerances.to_csv(tolerance_path, index=False)

    plot_combined_convergence(diagnostics, args.output_dir, args.formats, domain_average)
    plot_final_ci_width_distribution(
        combined,
        args.output_dir,
        args.formats,
        args.ci_reference_lines,
    )
    plot_window_stability(diagnostics, args.output_dir, args.formats, args.window_size)

    rank_df = rank_stability_table(combined, args.rank_top_k)
    rank_path = args.output_dir / f"combined_{max_permutations}_rank_stability.csv"
    rank_df.to_csv(rank_path, index=False)
    plot_rank_stability(rank_df, args.output_dir, args.formats, args.rank_top_k)

    checkpoint_df = checkpoint_sensitivity_table(rank_df, args.checkpoints, args.rank_top_k)
    checkpoint_path = args.output_dir / f"combined_{max_permutations}_checkpoint_sensitivity.csv"
    checkpoint_df.to_csv(checkpoint_path, index=False)
    plot_checkpoint_sensitivity(checkpoint_df, args.output_dir, args.formats, args.rank_top_k)

    plot_top_expert_cis(combined, args.output_dir, args.formats, args.top_experts)

    if args.plot_trajectories:
        plot_combined_trajectories(combined, args.output_dir, args.formats, args.top_k)

    print(f"Saved combined running contributions to {combined_path}")
    print(f"Saved combined convergence diagnostics to {diagnostics_path}")
    print(f"Saved metric-specific tolerances to {tolerance_path}")
    print(f"Saved rank stability to {rank_path}")
    print(f"Saved checkpoint sensitivity to {checkpoint_path}")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
