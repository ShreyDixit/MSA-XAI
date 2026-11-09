from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from msapy.datastructures import ShapleyModeND
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests


BASE_DIR = Path(__file__).resolve().parent
SHAPLEY_MODES_PATH = BASE_DIR / "shapley_modes"
FIGURE_PATH = BASE_DIR / "figures"
STATISTICS_PATH = BASE_DIR / "statistics"

TYPES_OF_REGULARISATIONS = [
    "l1 regularization",
    "l2 regularization",
    "none regularization",
]

REGULARISATION_LABELS = {
    "l1 regularization": "L1",
    "l2 regularization": "L2",
    "none regularization": "None",
}

CM = 1 / 2.54
PURPLE = "#A6587C"
NIGHT_BLUE = "#394D73"
GRAY = "#b8b8b8"
HALF_BLACK = "#232324"
REGULARISATION_COLORS = [PURPLE, NIGHT_BLUE, GRAY]
STAT_ALPHA = 0.05


def set_visual_style():
    sns.set_theme(
        style="ticks",
        font_scale=0.75,
        rc={
            "font.family": "DejaVu Sans",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.labelpad": 2,
            "axes.linewidth": 0.5,
            "axes.titlepad": 4,
            "lines.linewidth": 1,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 2,
            "xtick.major.pad": 1,
            "xtick.major.width": 0.5,
            "ytick.major.size": 2,
            "ytick.major.pad": 1,
            "ytick.major.width": 0.5,
            "text.color": HALF_BLACK,
            "axes.edgecolor": HALF_BLACK,
            "axes.labelcolor": HALF_BLACK,
            "xtick.color": HALF_BLACK,
            "ytick.color": HALF_BLACK,
        },
    )


def index_of_distributed_computation(contributions):
    """
    Compute the entropy-based IODC metric for signed neuron contributions.
    """
    contributions = np.asarray(contributions, dtype=float)
    s = contributions**2
    total = np.sum(s)

    if total == 0:
        return 0.0

    p = s / total
    p = np.where(p == 0, 1e-10, p)

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(contributions))
    iodc = entropy / max_entropy if max_entropy != 0 else 0.0

    return max(0.0, min(1.0, iodc))


def regularisation_label(regularisation):
    return REGULARISATION_LABELS[regularisation]


def trial_index_from_path(path):
    return int(path.stem.split("_")[-1])


def load_iodc_dataframe():
    rows = []

    for regularisation in TYPES_OF_REGULARISATIONS:
        regularisation_path = SHAPLEY_MODES_PATH / regularisation
        trial_paths = sorted(regularisation_path.glob("trial_*.csv"), key=trial_index_from_path)

        if not trial_paths:
            raise FileNotFoundError(f"No trial CSV files found in {regularisation_path}")

        for trial_path in trial_paths:
            shapley_modes = ShapleyModeND.from_csv(trial_path)
            rows.append(
                {
                    "regularisation": regularisation,
                    "regularisation_label": regularisation_label(regularisation),
                    "trial_idx": trial_index_from_path(trial_path),
                    "iodc": index_of_distributed_computation(shapley_modes.mean()),
                }
            )

    return pd.DataFrame(rows)


def add_fdr_correction(df, p_col="p_value", out_col="p_fdr", alpha=STAT_ALPHA):
    df = df.copy()
    df[out_col] = np.nan

    p_values = df[p_col].astype(float)
    finite_mask = np.isfinite(p_values)

    if finite_mask.any():
        df.loc[finite_mask, out_col] = multipletests(
            p_values.loc[finite_mask],
            method="fdr_bh",
        )[1]

    df[f"significant_{out_col}"] = np.where(df[out_col] < alpha, "yes", "no")
    return df


def summarize_iodc(df_iodc):
    return (
        df_iodc.groupby("regularisation_label")
        .agg(
            n=("iodc", "count"),
            mean_iodc=("iodc", "mean"),
            std_iodc=("iodc", "std"),
            median_iodc=("iodc", "median"),
            min_iodc=("iodc", "min"),
            max_iodc=("iodc", "max"),
        )
        .reindex([regularisation_label(r) for r in TYPES_OF_REGULARISATIONS])
        .reset_index()
    )


def run_global_test(df_iodc):
    groups = [
        df_iodc[df_iodc["regularisation"] == regularisation]["iodc"].dropna().to_numpy()
        for regularisation in TYPES_OF_REGULARISATIONS
    ]

    total_n = sum(len(group) for group in groups)
    num_groups = len(groups)

    try:
        statistic, p_value = kruskal(*groups)
    except ValueError:
        statistic, p_value = np.nan, 1.0

    epsilon_squared = (
        (statistic - num_groups + 1) / (total_n - num_groups)
        if total_n > num_groups and np.isfinite(statistic)
        else np.nan
    )
    epsilon_squared = max(0.0, epsilon_squared) if np.isfinite(epsilon_squared) else np.nan

    return pd.DataFrame(
        [
            {
                "test": "Kruskal-Wallis",
                "n_total": total_n,
                "h_statistic": statistic,
                "epsilon_squared": epsilon_squared,
                "p_value": p_value,
            }
        ]
    )


def run_pairwise_tests(df_iodc):
    rows = []

    for regularisation_1, regularisation_2 in combinations(TYPES_OF_REGULARISATIONS, 2):
        values_1 = df_iodc[df_iodc["regularisation"] == regularisation_1]["iodc"].dropna().to_numpy()
        values_2 = df_iodc[df_iodc["regularisation"] == regularisation_2]["iodc"].dropna().to_numpy()

        statistic, p_value = mannwhitneyu(values_1, values_2, alternative="two-sided")
        rank_biserial = (2 * statistic / (len(values_1) * len(values_2))) - 1

        rows.append(
            {
                "comparison": f"{regularisation_label(regularisation_1)} vs {regularisation_label(regularisation_2)}",
                "n_1": len(values_1),
                "n_2": len(values_2),
                "median_1": np.median(values_1),
                "median_2": np.median(values_2),
                "median_diff_1_minus_2": np.median(values_1) - np.median(values_2),
                "u_statistic": statistic,
                "rank_biserial_1_vs_2": rank_biserial,
                "p_value": p_value,
            }
        )

    return add_fdr_correction(pd.DataFrame(rows))


def print_table(title, df):
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


def p_to_stars(p_value):
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_significance_bracket(ax, x1, x2, y, height, label):
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=0.6, c=HALF_BLACK)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", fontsize=7)


def plot_iodc_boxplot(df_iodc, df_pairwise):
    order = [regularisation_label(r) for r in TYPES_OF_REGULARISATIONS]
    fig, ax = plt.subplots(figsize=(9 * CM, 7 * CM), dpi=600)

    color_by_regularisation = dict(zip(order, REGULARISATION_COLORS))

    sns.boxplot(
        data=df_iodc,
        x="regularisation_label",
        y="iodc",
        hue="regularisation_label",
        order=order,
        hue_order=order,
        palette=color_by_regularisation,
        width=0.55,
        fliersize=0,
        linewidth=0.8,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_iodc,
        x="regularisation_label",
        y="iodc",
        order=order,
        color=HALF_BLACK,
        size=2.2,
        jitter=0.16,
        alpha=0.65,
        ax=ax,
    )

    y_min = df_iodc["iodc"].min()
    y_max = df_iodc["iodc"].max()
    y_range = max(y_max - y_min, 0.02)
    bracket_height = y_range * 0.035
    bracket_gap = y_range * 0.11
    start_y = y_max + y_range * 0.08

    comparison_positions = {
        "L1 vs L2": (0, 1),
        "L1 vs None": (0, 2),
        "L2 vs None": (1, 2),
    }

    for idx, comparison in enumerate(comparison_positions):
        p_value = df_pairwise.loc[df_pairwise["comparison"] == comparison, "p_fdr"].iloc[0]
        x1, x2 = comparison_positions[comparison]
        add_significance_bracket(
            ax,
            x1,
            x2,
            start_y + idx * bracket_gap,
            bracket_height,
            p_to_stars(p_value),
        )

    ax.set_xlabel("Regularization")
    ax.set_ylabel("Index of Distributed Computation")
    ax.set_ylim(y_min - y_range * 0.08, start_y + 3 * bracket_gap)
    ax.set_title("CNN Regularization Effect on Distributed Computation")
    sns.despine(ax=ax, offset=1, trim=True)
    fig.tight_layout(pad=0.4)

    FIGURE_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_PATH / "CNN Regularization IODC Boxplot.pdf",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_PATH / "CNN Regularization IODC Boxplot.png",
        dpi=600,
        bbox_inches="tight",
    )


def main():
    set_visual_style()
    STATISTICS_PATH.mkdir(parents=True, exist_ok=True)

    df_iodc = load_iodc_dataframe()
    df_summary = summarize_iodc(df_iodc)
    df_global = add_fdr_correction(run_global_test(df_iodc))
    df_pairwise = run_pairwise_tests(df_iodc)

    df_iodc.to_csv(STATISTICS_PATH / "cnn_iodc_values.csv", index=False)
    df_summary.to_csv(STATISTICS_PATH / "cnn_iodc_summary.csv", index=False)
    df_global.to_csv(STATISTICS_PATH / "cnn_iodc_global_test.csv", index=False)
    df_pairwise.to_csv(STATISTICS_PATH / "cnn_iodc_pairwise_tests.csv", index=False)

    print("\n" + "=" * 100)
    print("CNN IODC STATISTICAL ANALYSIS")
    print("=" * 100)
    print(f"Alpha: {STAT_ALPHA}")
    print("Multiple-comparison correction: Benjamini-Hochberg FDR.")
    print("Trial handling: independent trials; trial_idx is not treated as a matched/repeated factor.")

    print_table("IODC summary by regularization", df_summary)
    print_table("IODC global test: L1 vs L2 vs None", df_global)
    print_table("IODC pairwise post-hoc tests: Mann-Whitney U", df_pairwise)

    plot_iodc_boxplot(df_iodc, df_pairwise)


if __name__ == "__main__":
    main()
