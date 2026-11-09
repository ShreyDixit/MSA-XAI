# %%
%load_ext autoreload
%autoreload 2

# %%
import numpy as np
import pandas as pd
import torch
from msapy.utils import distribution_of_processing
from data import get_data_loaders
from msapy.datastructures import ShapleyModeND
from dataclasses import dataclass
from model import MNISTNet
from typing import Optional
import os
from evaluation import evaluate_model

import seaborn as sns
import matplotlib.pyplot as plt
from visual_config import *
from scipy.stats import pearsonr

# %%
device = torch.device("cpu")
num_permutations = 1000
num_neurons = 100
num_epochs = 6
num_trials = 10

models_path = "models"
shapley_modes_path = "shapley_modes"
figure_path = "figures"

# %%
train_loader, test_loader = get_data_loaders(device)

# %%
@dataclass
class ShapleyResult:
    shapley_modes: ShapleyModeND
    model: MNISTNet
    regularisation: str
    num_neurons: int
    trial_idx: int

    @classmethod
    def from_files(cls, regularisation, num_neurons, trial_idx):
        shapley_modes = ShapleyModeND.from_csv(
           os.path.join(shapley_modes_path, regularisation, str(num_neurons), f"trial_{trial_idx}.csv")
        )
        model = torch.load(os.path.join(models_path, regularisation, str(num_neurons), f"trial_{trial_idx}.pt"), map_location=device)

        return cls(shapley_modes, model, regularisation, num_neurons, trial_idx)

class ShapleyResultManager:
    def __init__(self):
        self.results = []

    def add_result(self, result: ShapleyResult):
        self.results.append(result)

    def get_results(self, regularisation: Optional[str] = None, num_neurons: Optional[int] = None) -> list[ShapleyResult]:
        results = self.results
        if regularisation is not None:
            results = [result for result in results if result.regularisation == regularisation]
        if num_neurons is not None:
            results = [result for result in results if result.num_neurons == num_neurons]
        return results

# %%
types_of_regularisations =  ["l1 regularization", "l2 regularization", "none regularization"]
num_neurons_list = sorted([int(n) for n in os.listdir(f"{models_path}/{types_of_regularisations[0]}")])

# %%
manager = ShapleyResultManager()
for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate(num_neurons_list):
        for trial_idx in range(num_trials):
            shapley_result = ShapleyResult.from_files(regularisation, num_neurons, trial_idx)
            manager.add_result(shapley_result)

# %%
l1_regularisation_results = manager.get_results(regularisation="l1 regularization")
l2_regularisation_results = manager.get_results(regularisation="l2 regularization")
none_regularisation_results = manager.get_results(regularisation="none regularization")

# %%
df_accuracy_drop = {
    "routine": [],
    "Accuracy": [],
    "Rank": [],
}

for i, regularisation in enumerate(types_of_regularisations):
    shapley_results = manager.get_results(regularisation=regularisation, num_neurons=200)
    for j, shapley_result in enumerate(shapley_results):
        top_40_neurons = list(shapley_result.shapley_modes.values.mean(0).argsort()[-40:])
        least_160_neurons = list(shapley_result.shapley_modes.values.mean(0).argsort()[:160])
        accuracy_intact = evaluate_model(shapley_result.model, test_loader)
        accuracy_after_dropping_top_40 = evaluate_model(shapley_result.model, test_loader, top_40_neurons)
        accuracy_after_dropping_least_160 = evaluate_model(shapley_result.model, test_loader, least_160_neurons)

        df_accuracy_drop["routine"].append(regularisation.split()[0].capitalize())
        df_accuracy_drop["Accuracy"].append(accuracy_after_dropping_least_160)
        df_accuracy_drop["Rank"].append("The Rest")

        df_accuracy_drop["routine"].append(regularisation.split()[0].capitalize())
        df_accuracy_drop["Accuracy"].append(accuracy_after_dropping_top_40)
        df_accuracy_drop["Rank"].append("Top 40")

        df_accuracy_drop["routine"].append(regularisation.split()[0].capitalize())
        df_accuracy_drop["Accuracy"].append(accuracy_intact)
        df_accuracy_drop["Rank"].append("No Lesion")

# %%
df = pd.DataFrame(df_accuracy_drop)
plt.figure(figsize=(10*CM,10*CM), dpi = 300)
sns.stripplot(data=df[df.Rank != "No Lesion"], hue="routine", y="Accuracy", x="Rank", palette=[PURPLE,NIGHT_BLUE,GRAY],size=3.5)
plt.xticks(ticks=[0,1],labels=["The Rest","Top 40"],rotation=0)
plt.title("Lesion Effect based on Rank")
plt.legend()
sns.despine(trim=False)
plt.tight_layout(pad=0.2)
#plt.savefig(f"figures/Lesion based on rank.pdf",dpi=600,bbox_inches='tight')

# %%
attribution_techniques = {
    "Shapley": shapley_modes_path,
    "IG": "ig_class_specific_signed",
    "Single Neuron": "single_neuron_ablation_class_specific_signed",
}


def load_attribution_scores(technique_path, regularisation, num_neurons, trial_idx):
    attribution_modes = ShapleyModeND.from_csv(
        os.path.join(technique_path, regularisation, str(num_neurons), f"trial_{trial_idx}.csv")
    )
    return attribution_modes.values.mean(0)


df_attribution_lesions = {
    "Attribution Technique": [],
    "Regularization": [],
    "Accuracy": [],
    "Lesion": [],
}

for regularisation in types_of_regularisations:
    for trial_idx in range(num_trials):
        model = torch.load(
            os.path.join(models_path, regularisation, "200", f"trial_{trial_idx}.pt"),
            map_location=device,
        )
        for attribution_name, attribution_path in attribution_techniques.items():
            contribution_scores = load_attribution_scores(attribution_path, regularisation, 200, trial_idx)
            ranked_neurons = contribution_scores.argsort()
            lesion_groups = {
                "Rest 160": list(ranked_neurons[:160]),
                "Top 40": list(ranked_neurons[-40:]),
            }

            for lesion_name, lesioned_neurons in lesion_groups.items():
                df_attribution_lesions["Attribution Technique"].append(attribution_name)
                df_attribution_lesions["Regularization"].append(regularisation.split()[0].capitalize())
                df_attribution_lesions["Accuracy"].append(evaluate_model(model, test_loader, lesioned_neurons))
                df_attribution_lesions["Lesion"].append(lesion_name)

# %%
df_attribution_lesions = pd.DataFrame(df_attribution_lesions)
regularization_palette = [PURPLE, NIGHT_BLUE, GRAY]

for lesion_name, output_name in [
    ("Top 40", "Lesion technique comparison Top 40.pdf"),
    ("Rest 160", "Lesion technique comparison Rest 160.pdf"),
]:
    plot_df = df_attribution_lesions[df_attribution_lesions["Lesion"] == lesion_name]

    plt.figure(figsize=(12 * CM, 10 * CM), dpi=300)
    sns.stripplot(
        data=plot_df,
        x="Attribution Technique",
        y="Accuracy",
        hue="Regularization",
        palette=regularization_palette,
        size=3.5,
        dodge=True,
        alpha=0.6,
    )
    sns.pointplot(
        data=plot_df,
        x="Attribution Technique",
        y="Accuracy",
        hue="Regularization",
        palette=regularization_palette,
        estimator=np.mean,
        errorbar=None,
        dodge=0.5,
        markers="_",
        markersize=14,
        linestyle="none",
        legend=False,
    )
    plt.title(f"{lesion_name} Lesion Effect by Attribution Technique")
    plt.xlabel("Attribution Technique")
    plt.ylabel("Accuracy")
    plt.legend(title="Regularization")
    sns.despine(trim=False)
    plt.tight_layout(pad=0.2)
    plt.savefig(os.path.join(figure_path, output_name), dpi=600, bbox_inches="tight")

# %%
def index_of_distributed_computation(contributions):
    """
    Computes the entropy-based metric for neuron contributions.

    Parameters:
    contributions (numpy.ndarray): Array of neuron contributions (can be negative).

    Returns:
    float: Entropy-based metric ranging between 0 and 1.
    """
    # Step 1: Calculate squared contributions
    s = contributions ** 2

    # Step 2: Compute total squared contributions
    S = np.sum(s)

    # Handle the case where S is zero to avoid division by zero
    if S == 0:
        return 0.0

    # Step 3: Compute normalized contributions
    p = s / S

    # Replace zeros in p with a small epsilon to avoid log(0)
    epsilon = 1e-10
    p = np.where(p == 0, epsilon, p)

    # Step 4: Compute entropy
    H = -np.sum(p * np.log(p))

    # Step 5: Compute maximum possible entropy
    N = len(contributions)
    H_max = np.log(N)

    # Step 6: Normalize the entropy to get the metric between 0 and 1
    M = H / H_max if H_max != 0 else 0.0

    # Ensure M is between 0 and 1
    M = max(0.0, min(1.0, M))

    return M


# %%
regularisation_iodc = np.zeros((len(types_of_regularisations), len(num_neurons_list), num_trials))

for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate(num_neurons_list):
        shapley_results = manager.get_results(regularisation=regularisation, num_neurons=num_neurons)
        for k, shapley_result in enumerate(shapley_results):
            regularisation_iodc[i, j, k] = index_of_distributed_computation(shapley_result.shapley_modes.mean())

# %%
plt.figure(figsize=(20*CM,10*CM),dpi = 600)
plt.errorbar(x=num_neurons_list, y=regularisation_iodc[0].mean(1), fmt='.', label="L1", yerr=regularisation_iodc[0].std(1), capsize=1,color = PURPLE)
plt.errorbar(x=num_neurons_list, y=regularisation_iodc[1].mean(1), fmt='.', label="L2", yerr=regularisation_iodc[1].std(1), capsize=1, color = NIGHT_BLUE)
plt.errorbar(x=num_neurons_list, y=regularisation_iodc[2].mean(1), fmt='.', label="None", yerr=regularisation_iodc[2].std(1), capsize=1, color = GRAY)
plt.xlabel("Number of Neurons")
plt.ylabel("Index of Distributed Computation")

plt.legend(bbox_to_anchor=(0.95, 1),frameon=False,)
sns.despine(offset=1, trim=True)
plt.tight_layout(pad=1)
plt.savefig(f"figures/Regularization Effect on Distributed Computation.pdf",dpi=600,bbox_inches='tight')

# %%
l1_reg_shapley_modes = manager.get_results(regularisation="l1 regularization", num_neurons=105)[0].shapley_modes
l2_reg_shapley_modes = manager.get_results(regularisation="l2 regularization", num_neurons=105)[0].shapley_modes
no_reg_shapley_modes = manager.get_results(regularisation="none regularization", num_neurons=105)[0].shapley_modes

# %%
fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[4,4, 4,0.2]), dpi=600, figsize=(18*CM, 6*CM))
fig.suptitle("Task Specific Contributions")

vmin = min(l1_reg_shapley_modes.values.min(), l2_reg_shapley_modes.values.min(), no_reg_shapley_modes.values.min())
vmax = max(l1_reg_shapley_modes.values.max(), l2_reg_shapley_modes.values.max(), no_reg_shapley_modes.values.max())

sns.heatmap(l1_reg_shapley_modes, annot=False, cbar=False, ax=axs[0], vmin=vmin, vmax=vmax, cmap=diverge_from_white,center=0)
sns.heatmap(l2_reg_shapley_modes, annot=False, yticklabels=False, cbar=False, ax=axs[1], vmax=vmax, vmin=vmin, cmap=diverge_from_white,center=0)
sns.heatmap(no_reg_shapley_modes, annot=False, yticklabels=False, cbar=False, ax=axs[2], vmax=vmax, vmin=vmin, cmap=diverge_from_white,center=0)

fig.colorbar(axs[2].collections[0], cax=axs[3])
axs[0].set_ylabel("MNIST Digits")
axs[0].set_title("L1")
axs[1].set_title("L2")
axs[2].set_title("None")
axs[1].set_xlabel("Neuron Index")

sns.despine(fig=fig, top=False, right=False, left=False, bottom=False)

fig.tight_layout(pad=0.2)
plt.savefig(f"figures/Digit Wise Shapley Values.pdf",dpi=600,bbox_inches='tight')

# %%
fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[4,4, 4,0.2]), dpi=600, figsize=(18*CM, 6*CM))
fig.suptitle("Correlation among Tasks")


correlation_l1 = np.corrcoef(l1_reg_shapley_modes.values, rowvar=True)
correlation_l2 = np.corrcoef(l2_reg_shapley_modes.values, rowvar=True)
correlation_no = np.corrcoef(no_reg_shapley_modes.values, rowvar=True)


sns.heatmap(correlation_l1, annot=False, cbar=False, ax=axs[0], vmin=-1, vmax=1, cmap=diverge_from_white,center=0)
sns.heatmap(correlation_l2, annot=False, yticklabels=False, cbar=False, ax=axs[1], vmax=1, vmin=-1, cmap=diverge_from_white,center=0)
sns.heatmap(correlation_no, annot=False, yticklabels=False, cbar=False, ax=axs[2], vmax=1, vmin=-1, cmap=diverge_from_white,center=0)

fig.colorbar(axs[2].collections[0], cax=axs[3])
axs[0].set_ylabel("MNIST Digits")
axs[0].set_title("L1")
axs[1].set_title("L2")
axs[2].set_title("None")
axs[1].set_xlabel("MNIST Digits")

sns.despine(fig=fig, top=False, right=False, left=False, bottom=False)

fig.tight_layout(pad=0.2)
plt.savefig(f"figures/Digit Wise Correlation.pdf",dpi=600,bbox_inches='tight')

# %%
correlation_matrix = np.zeros((len(types_of_regularisations), len(num_neurons_list), num_trials))

for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate(num_neurons_list):
        shapley_results = manager.get_results(regularisation=regularisation, num_neurons=num_neurons)
        for k, shapley_result in enumerate(shapley_results):
            outgoing_weights = shapley_result.model.layer2[1].weight.abs().mean(0).detach().cpu().numpy()
            incoming_weights = shapley_result.model.layer1.weight.abs().mean(1).detach().cpu().numpy()
            average_weights = (outgoing_weights + incoming_weights) / 2
            contributions = shapley_result.shapley_modes.values.mean(0)
            correlation_matrix[i, j, k] = pearsonr(average_weights, contributions)[0]

# %%
plt.figure(figsize=(20*CM,10*CM),dpi = 600)
plt.errorbar(x=num_neurons_list, y=correlation_matrix[0].mean(1), fmt='.', label="L1", yerr=correlation_matrix[0].std(1), capsize=1,color = PURPLE)
plt.errorbar(x=num_neurons_list, y=correlation_matrix[1].mean(1), fmt='.', label="L2", yerr=correlation_matrix[1].std(1), capsize=1, color = NIGHT_BLUE)
plt.errorbar(x=num_neurons_list, y=correlation_matrix[2].mean(1), fmt='.', label="None", yerr=correlation_matrix[2].std(1), capsize=1, color = GRAY)
plt.xlabel("Number of Neurons")
plt.ylabel("Correlation b/w Contributions and Weights")

plt.legend(bbox_to_anchor=(0.95, 1),frameon=False,)
sns.despine(offset=1, trim=True)
plt.tight_layout(pad=1)

plt.savefig(f"figures/Correlation between Weights and Contributions.pdf",dpi=600,bbox_inches='tight')

# %%
digit_wise_iodc = np.zeros((3, len(num_neurons_list), num_trials))

digits = [0, 1, 9]

for i, digit in enumerate(digits):
    for j, num_neurons in enumerate(num_neurons_list):
        shapley_results = manager.get_results(regularisation="l2 regularization", num_neurons=num_neurons)
        for k, shapley_result in enumerate(shapley_results):
            digit_wise_iodc[i, j, k] = index_of_distributed_computation(shapley_result.shapley_modes.iloc[digit])

# %%
plt.figure(figsize=(20*CM,10*CM),dpi = 600)
COLORS = [PURPLE, NIGHT_BLUE, GRAY]

for i, digit in enumerate(digits):
    plt.errorbar(x=num_neurons_list, y=digit_wise_iodc[i].mean(1), fmt='.', label=str(digit), yerr=digit_wise_iodc[i].std(1), capsize=1, color = COLORS[i])
plt.xlabel("Number of Neurons")
plt.ylabel("Index of Distributed Computation")

plt.legend(bbox_to_anchor=(0.95, 1),frameon=False,)
sns.despine(offset=1, trim=True)
plt.tight_layout(pad=1)
plt.savefig(f"figures/Index of Distributed Computation Digit Wise.pdf",dpi=600,bbox_inches='tight')

# %%
num_neurons_list = sorted(num_neurons_list)

# %%
import matplotlib.pyplot as plt

rows, cols = 6, 6
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    # Replace the following with your actual plotting logic
    relevant_results = manager.get_results(regularisation="none regularization", num_neurons=num_neurons_list[3:-1][i])
    incoming_weights = relevant_results[0].model.layer1.weight.abs().mean(1).detach().cpu().numpy()
    outgoing_weights = relevant_results[0].model.layer2[1].weight.abs().mean(0).detach().cpu().numpy()
    average_weights = (outgoing_weights + incoming_weights) / 2

    contributions = relevant_results[0].shapley_modes.values.mean(0)
    
    # Scatter plot with circular, translucent dots
    ax.scatter(average_weights, contributions, alpha=0.6, s=40, edgecolor="k")  # s adjusts the size of dots
    ax.set_title(f'{num_neurons_list[3:-1][i]} Neurons', fontsize=8)
    ax.axis('off')  # Optional: Turn off individual axes for cleaner look

# Overall figure title and axis labels
fig.suptitle("Neural Network Weight and Contribution Analysis", fontsize=16, fontweight='bold')
fig.text(0.5, 0.04, "Average Weights", ha='center', fontsize=12)
fig.text(0.04, 0.5, "Shapley Contributions", va='center', rotation='vertical', fontsize=12)

plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.5)  # Adjust layout for better spacing
plt.show()


# %%

rows, cols = 6, 6
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    # Replace the following with your actual plotting logic
    relevant_results = manager.get_results(regularisation="l1 regularization", num_neurons=num_neurons_list[3:-1][i])
    incoming_weights = relevant_results[0].model.layer1.weight.abs().mean(1).detach().cpu().numpy()
    outgoing_weights = relevant_results[0].model.layer2[1].weight.abs().mean(0).detach().cpu().numpy()
    average_weights = (outgoing_weights + incoming_weights) / 2

    contributions = relevant_results[0].shapley_modes.values.mean(0)
    
    # Scatter plot with circular, translucent dots
    ax.scatter(average_weights, contributions, alpha=0.6, s=40, edgecolor="k")  # s adjusts the size of dots
    ax.set_title(f'{num_neurons_list[3:-1][i]} Neurons', fontsize=8)
    ax.axis('off')  # Optional: Turn off individual axes for cleaner look

# Overall figure title and axis labels
fig.suptitle("Neural Network Weight and Contribution Analysis", fontsize=16, fontweight='bold')
fig.text(0.5, 0.04, "Average Weights", ha='center', fontsize=12)
fig.text(0.04, 0.5, "Shapley Contributions", va='center', rotation='vertical', fontsize=12)

plt.subplots_adjust(top=0.92, wspace=0.3, hspace=0.5)  # Adjust layout for better spacing
plt.show()


# %%
shapley_modes_path = "shapley_modes_mean"

manager_mean = ShapleyResultManager()
for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate([200]):
        for trial_idx in range(num_trials):
            shapley_result = ShapleyResult.from_files(regularisation, num_neurons, trial_idx)
            manager_mean.add_result(shapley_result)

# %%
mean_zero_similarity_rows = []

for regularisation in types_of_regularisations:
    zero_results = sorted(
        manager.get_results(regularisation=regularisation, num_neurons=200),
        key=lambda result: result.trial_idx,
    )
    mean_results = sorted(
        manager_mean.get_results(regularisation=regularisation, num_neurons=200),
        key=lambda result: result.trial_idx,
    )

    if len(zero_results) != num_trials or len(mean_results) != num_trials:
        raise ValueError(
            f"Expected {num_trials} zero and mean perturbation results for "
            f"{regularisation} at 200 neurons, but found "
            f"{len(zero_results)} zero and {len(mean_results)} mean results."
        )

    for zero_result, mean_result in zip(zero_results, mean_results):
        if zero_result.trial_idx != mean_result.trial_idx:
            raise ValueError(
                f"Trial mismatch for {regularisation}: "
                f"zero trial {zero_result.trial_idx}, mean trial {mean_result.trial_idx}."
            )

        zero_values = zero_result.shapley_modes.values.flatten()
        mean_values = mean_result.shapley_modes.values.flatten()
        correlation, p_value = pearsonr(zero_values, mean_values)

        mean_zero_similarity_rows.append({
            "Regularisation": regularisation.split()[0].capitalize(),
            "Trial": zero_result.trial_idx,
            "Pearson r": correlation,
            "p_value": p_value,
        })

df_mean_zero_similarity = pd.DataFrame(mean_zero_similarity_rows)

mean_zero_similarity_summary = (
    df_mean_zero_similarity
    .groupby("Regularisation")
    .agg(
        n_trials=("Pearson r", "count"),
        mean_r=("Pearson r", "mean"),
        std_r=("Pearson r", "std"),
        min_r=("Pearson r", "min"),
        max_r=("Pearson r", "max"),
    )
    .reset_index()
)

print("\nMean-vs-zero perturbation Shapley similarity at 200 neurons")
print(mean_zero_similarity_summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))

plt.figure(figsize=(8*CM, 8*CM), dpi=600)
regularisation_plot_order = [
    regularisation.split()[0].capitalize()
    for regularisation in types_of_regularisations
]
sns.stripplot(
    data=df_mean_zero_similarity,
    x="Regularisation",
    y="Pearson r",
    order=regularisation_plot_order,
    palette=[PURPLE, NIGHT_BLUE, GRAY],
    size=4,
    jitter=0.18,
)

for x_idx, regularisation in enumerate(regularisation_plot_order):
    values = df_mean_zero_similarity[
        df_mean_zero_similarity["Regularisation"] == regularisation
    ]["Pearson r"]
    plt.errorbar(
        x=x_idx,
        y=values.mean(),
        yerr=values.std(),
        fmt="_",
        color="black",
        markersize=16,
        capsize=3,
        linewidth=1,
    )

plt.ylim(0, 1.02)
plt.ylabel("Pearson correlation")
plt.xlabel("")
plt.title("Mean vs Zero Perturbation Similarity")
sns.despine(trim=True)
plt.tight_layout(pad=0.5)
plt.savefig(
    f"{figure_path}/Mean vs Zero Perturbation Similarity.pdf",
    dpi=600,
    bbox_inches="tight",
)

# %%


# %%
# Statistical analysis for IODC and weight-Shapley correlation plots.
#
# Trial indices are treated as independent because trials are not matched across
# regularisations or across neuron counts.
from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_1samp
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

STAT_ALPHA = 0.05


def regularisation_label(regularisation):
    return regularisation.split()[0].capitalize()


def print_table(title, df):
    print(f"\n{title}")
    if df.empty:
        print("No rows.")
        return
    print(df.to_string(index=False, float_format=lambda value: f"{value:.6g}"))


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


def add_grouped_fdr_correction(df, group_cols, p_col="p_value", out_col="p_fdr", alpha=STAT_ALPHA):
    df = df.copy()
    df[out_col] = np.nan

    for _, indices in df.groupby(group_cols).groups.items():
        p_values = df.loc[indices, p_col].astype(float)
        finite_mask = np.isfinite(p_values)

        if finite_mask.any():
            corrected = multipletests(
                p_values.loc[finite_mask],
                method="fdr_bh",
            )[1]
            df.loc[p_values.loc[finite_mask].index, out_col] = corrected

    df[f"significant_{out_col}"] = np.where(df[out_col] < alpha, "yes", "no")
    return df


def safe_pearsonr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 2 or np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return np.nan, np.nan

    return pearsonr(x, y)


def safe_spearmanr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 2 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan

    statistic, p_value = spearmanr(x, y)
    return statistic, p_value


def safe_ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(x) < 2 or np.isclose(np.std(x), 0.0):
        return {
            "n": len(x),
            "intercept": np.nan,
            "slope": np.nan,
            "slope_p_value": np.nan,
            "r_squared": np.nan,
        }

    x_df = pd.DataFrame({"num_neurons": x})
    x_df = sm.add_constant(x_df)
    model = sm.OLS(y, x_df).fit()

    return {
        "n": int(model.nobs),
        "intercept": model.params["const"],
        "slope": model.params["num_neurons"],
        "slope_p_value": model.pvalues["num_neurons"],
        "r_squared": model.rsquared,
    }


def safe_one_sample_ttest(values, popmean=0.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan, np.nan

    sample_std = np.std(values, ddof=1)
    sample_mean = np.mean(values)

    if np.isclose(sample_std, 0.0):
        if np.isclose(sample_mean, popmean):
            return 0.0, 1.0
        return np.inf if sample_mean > popmean else -np.inf, 0.0

    statistic, p_value = ttest_1samp(values, popmean=popmean)
    return statistic, p_value


def fisher_z_transform(r_values):
    r_values = np.asarray(r_values, dtype=float)
    return np.arctanh(np.clip(r_values, -0.999999999999, 0.999999999999))


def tanh_nanmean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return np.tanh(np.mean(values))


print("\n" + "=" * 100)
print("STATISTICAL ANALYSIS")
print("=" * 100)
print(f"Alpha: {STAT_ALPHA}")
print("Multiple-comparison correction: Benjamini-Hochberg FDR.")
print("Trial handling: independent trials; trial_idx is not treated as a matched/repeated factor.")


# %%
# 1. IODC: compare regularisations separately at each neuron count.
iodc_rows = []

for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate(num_neurons_list):
        for trial_idx in range(num_trials):
            iodc_rows.append({
                "regularisation": regularisation,
                "regularisation_label": regularisation_label(regularisation),
                "num_neurons": num_neurons,
                "trial_idx": trial_idx,
                "iodc": regularisation_iodc[i, j, trial_idx],
            })

df_iodc_stats = pd.DataFrame(iodc_rows)

iodc_summary = (
    df_iodc_stats
    .groupby(["num_neurons", "regularisation_label"])
    .agg(
        n=("iodc", "count"),
        mean_iodc=("iodc", "mean"),
        std_iodc=("iodc", "std"),
        median_iodc=("iodc", "median"),
        min_iodc=("iodc", "min"),
        max_iodc=("iodc", "max"),
    )
    .reset_index()
)

print_table("IODC summary by neuron count and regularisation", iodc_summary)

iodc_kruskal_rows = []

for num_neurons in sorted(df_iodc_stats["num_neurons"].unique()):
    groups = []

    for regularisation in types_of_regularisations:
        values = df_iodc_stats[
            (df_iodc_stats["num_neurons"] == num_neurons) &
            (df_iodc_stats["regularisation"] == regularisation)
        ]["iodc"].dropna().to_numpy()
        groups.append(values)

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

    iodc_kruskal_rows.append({
        "num_neurons": num_neurons,
        "test": "Kruskal-Wallis",
        "n_total": total_n,
        "h_statistic": statistic,
        "epsilon_squared": epsilon_squared,
        "p_value": p_value,
    })

df_iodc_kruskal = add_fdr_correction(pd.DataFrame(iodc_kruskal_rows))
print_table(
    "IODC global test: L1 vs L2 vs None at each neuron count",
    df_iodc_kruskal,
)

iodc_pairwise_rows = []

for num_neurons in sorted(df_iodc_stats["num_neurons"].unique()):
    for regularisation_1, regularisation_2 in combinations(types_of_regularisations, 2):
        values_1 = df_iodc_stats[
            (df_iodc_stats["num_neurons"] == num_neurons) &
            (df_iodc_stats["regularisation"] == regularisation_1)
        ]["iodc"].dropna().to_numpy()

        values_2 = df_iodc_stats[
            (df_iodc_stats["num_neurons"] == num_neurons) &
            (df_iodc_stats["regularisation"] == regularisation_2)
        ]["iodc"].dropna().to_numpy()

        statistic, p_value = mannwhitneyu(values_1, values_2, alternative="two-sided")
        rank_biserial = (2 * statistic / (len(values_1) * len(values_2))) - 1

        iodc_pairwise_rows.append({
            "num_neurons": num_neurons,
            "comparison": f"{regularisation_label(regularisation_1)} vs {regularisation_label(regularisation_2)}",
            "n_1": len(values_1),
            "n_2": len(values_2),
            "median_1": np.median(values_1),
            "median_2": np.median(values_2),
            "median_diff_1_minus_2": np.median(values_1) - np.median(values_2),
            "u_statistic": statistic,
            "rank_biserial_1_vs_2": rank_biserial,
            "p_value": p_value,
        })

df_iodc_pairwise = add_fdr_correction(pd.DataFrame(iodc_pairwise_rows))
print_table(
    "IODC pairwise post-hoc tests: Mann-Whitney U at each neuron count",
    df_iodc_pairwise,
)


# %%
# 2. IODC: trend with number of neurons for each regularisation.
iodc_trend_rows = []
iodc_ols_rows = []

for regularisation in types_of_regularisations:
    df_regularisation = df_iodc_stats[df_iodc_stats["regularisation"] == regularisation]

    trend_inputs = {
        "all_trials": df_regularisation[["num_neurons", "iodc"]],
        "mean_per_neuron_count": (
            df_regularisation
            .groupby("num_neurons", as_index=False)["iodc"]
            .mean()
        ),
    }

    for data_scope, trend_df in trend_inputs.items():
        x = trend_df["num_neurons"].to_numpy(dtype=float)
        y = trend_df["iodc"].to_numpy(dtype=float)

        pearson_statistic, pearson_p = safe_pearsonr(x, y)
        spearman_statistic, spearman_p = safe_spearmanr(x, y)

        iodc_trend_rows.append({
            "regularisation": regularisation_label(regularisation),
            "data_scope": data_scope,
            "test": "Pearson",
            "n": len(trend_df),
            "statistic": pearson_statistic,
            "p_value": pearson_p,
        })

        iodc_trend_rows.append({
            "regularisation": regularisation_label(regularisation),
            "data_scope": data_scope,
            "test": "Spearman",
            "n": len(trend_df),
            "statistic": spearman_statistic,
            "p_value": spearman_p,
        })

        ols_result = safe_ols_slope(x, y)
        iodc_ols_rows.append({
            "regularisation": regularisation_label(regularisation),
            "data_scope": data_scope,
            **ols_result,
        })

df_iodc_trends = add_grouped_fdr_correction(
    pd.DataFrame(iodc_trend_rows),
    group_cols=["data_scope", "test"],
)
print_table(
    "IODC trend with number of neurons: Pearson and Spearman correlations",
    df_iodc_trends,
)

df_iodc_ols = add_grouped_fdr_correction(
    pd.DataFrame(iodc_ols_rows),
    group_cols=["data_scope"],
    p_col="slope_p_value",
    out_col="slope_p_fdr",
)
print_table(
    "IODC linear trend with number of neurons: OLS slope tests",
    df_iodc_ols,
)


# %%
# 3. Weight-Shapley correlations: test whether the mean trial correlation is
# different from zero for each regularisation and neuron count.
weight_corr_rows = []

for i, regularisation in enumerate(types_of_regularisations):
    for j, num_neurons in enumerate(num_neurons_list):
        for trial_idx in range(num_trials):
            r_value = correlation_matrix[i, j, trial_idx]
            weight_corr_rows.append({
                "regularisation": regularisation,
                "regularisation_label": regularisation_label(regularisation),
                "num_neurons": num_neurons,
                "trial_idx": trial_idx,
                "pearson_r": r_value,
                "fisher_z": fisher_z_transform(r_value),
            })

df_weight_corr_stats = pd.DataFrame(weight_corr_rows)

weight_corr_summary = (
    df_weight_corr_stats
    .groupby(["num_neurons", "regularisation_label"])
    .agg(
        n=("pearson_r", "count"),
        mean_r=("pearson_r", "mean"),
        std_r=("pearson_r", "std"),
        median_r=("pearson_r", "median"),
        mean_fisher_z=("fisher_z", "mean"),
        mean_r_from_fisher_z=("fisher_z", tanh_nanmean),
    )
    .reset_index()
)

print_table(
    "Weight-Shapley correlation summary by neuron count and regularisation",
    weight_corr_summary,
)

weight_corr_test_rows = []

for (num_neurons, regularisation_label_value), group in df_weight_corr_stats.groupby(["num_neurons", "regularisation_label"]):
    z_values = group["fisher_z"].dropna().to_numpy()
    statistic, p_value = safe_one_sample_ttest(z_values, popmean=0.0)

    weight_corr_test_rows.append({
        "num_neurons": num_neurons,
        "regularisation": regularisation_label_value,
        "n_trials": len(z_values),
        "mean_r": group["pearson_r"].mean(),
        "mean_fisher_z": np.mean(z_values) if len(z_values) else np.nan,
        "mean_r_from_fisher_z": tanh_nanmean(z_values),
        "t_statistic": statistic,
        "df": len(z_values) - 1,
        "p_value": p_value,
    })

df_weight_corr_tests = add_fdr_correction(pd.DataFrame(weight_corr_test_rows))
print_table(
    "Weight-Shapley correlation tests: one-sample t-test on Fisher-z values against zero",
    df_weight_corr_tests,
)


# %%
# Compact visualization of the statistical results.
from matplotlib.colors import TwoSlopeNorm


def format_p_value(p_value):
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.3f}"


def format_p_value_dataframe(df):
    return df.apply(lambda column: column.map(format_p_value))


def pvalue_pivot(df, index, columns, p_col, row_order=None, column_order=None):
    p_values = df.pivot(index=index, columns=columns, values=p_col)

    if row_order is not None:
        p_values = p_values.reindex(row_order)
    if column_order is not None:
        p_values = p_values.reindex(columns=column_order)

    return p_values.astype(float)


def plot_pvalue_heatmap(p_values, title, xlabel, ylabel, filename, figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=600)

    sns.heatmap(
        p_values,
        ax=ax,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=0.0, vcenter=STAT_ALPHA, vmax=1.0),
        annot=format_p_value_dataframe(p_values),
        fmt="",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={
            "label": "FDR-corrected p-value",
            "ticks": [0.0, STAT_ALPHA, 0.5, 1.0],
        },
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    apply_compact_heatmap_ticks(ax)

    fig.text(
        0.5,
        0.01,
        f"Blue: p < {STAT_ALPHA}; white: p = {STAT_ALPHA}; red: p > {STAT_ALPHA}.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1], pad=0.4)

    os.makedirs(figure_path, exist_ok=True)
    plt.savefig(
        f"{figure_path}/{filename}",
        dpi=600,
        bbox_inches="tight",
    )


def apply_compact_heatmap_ticks(ax):
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)


regularisation_order = [regularisation_label(r) for r in types_of_regularisations]
regularisation_colors = {
    regularisation_label(types_of_regularisations[0]): PURPLE,
    regularisation_label(types_of_regularisations[1]): NIGHT_BLUE,
    regularisation_label(types_of_regularisations[2]): GRAY,
}
comparison_order = [
    f"{regularisation_label(r1)} vs {regularisation_label(r2)}"
    for r1, r2 in combinations(types_of_regularisations, 2)
]
neuron_order = sorted(num_neurons_list)

# Figure 1: Global IODC regularisation test p-values.
iodc_global_pvalues = pvalue_pivot(
    df_iodc_kruskal.assign(test_label="Kruskal-Wallis"),
    index="test_label",
    columns="num_neurons",
    p_col="p_fdr",
    row_order=["Kruskal-Wallis"],
    column_order=neuron_order,
)
plot_pvalue_heatmap(
    iodc_global_pvalues,
    title="IODC Global Test P-values",
    xlabel="Number of Neurons",
    ylabel="",
    filename="IODC Global Test P-values.pdf",
    figsize=(30*CM, 5*CM),
)

# Figure 2: Pairwise post-hoc IODC test p-values.
iodc_pairwise_pvalues = pvalue_pivot(
    df_iodc_pairwise,
    index="comparison",
    columns="num_neurons",
    p_col="p_fdr",
    row_order=comparison_order,
    column_order=neuron_order,
)
plot_pvalue_heatmap(
    iodc_pairwise_pvalues,
    title="IODC Pairwise Mann-Whitney P-values",
    xlabel="Number of Neurons",
    ylabel="Comparison",
    filename="IODC Pairwise Mann-Whitney P-values.pdf",
    figsize=(30*CM, 8*CM),
)

# Figure 3: IODC trend p-values.
df_iodc_trends_plot = df_iodc_trends.copy()
df_iodc_trends_plot["trend_test"] = (
    df_iodc_trends_plot["data_scope"]
    .replace({
        "all_trials": "All trials",
        "mean_per_neuron_count": "Mean per neuron count",
    }) +
    " - " +
    df_iodc_trends_plot["test"]
)
trend_test_order = [
    "All trials - Spearman",
    "All trials - Pearson",
    "Mean per neuron count - Spearman",
    "Mean per neuron count - Pearson",
]
iodc_trend_pvalues = pvalue_pivot(
    df_iodc_trends_plot,
    index="regularisation",
    columns="trend_test",
    p_col="p_fdr",
    row_order=regularisation_order,
    column_order=trend_test_order,
)
plot_pvalue_heatmap(
    iodc_trend_pvalues,
    title="IODC Trend P-values",
    xlabel="Trend Test",
    ylabel="Regularisation",
    filename="IODC Trend P-values.pdf",
    figsize=(18*CM, 8*CM),
)

# Figure 4: Weight-Shapley correlation test p-values.
weight_corr_pvalues = pvalue_pivot(
    df_weight_corr_tests,
    index="regularisation",
    columns="num_neurons",
    p_col="p_fdr",
    row_order=regularisation_order,
    column_order=neuron_order,
)
plot_pvalue_heatmap(
    weight_corr_pvalues,
    title="Weight-Shapley Fisher-z Test P-values",
    xlabel="Number of Neurons",
    ylabel="Regularisation",
    filename="Weight-Shapley Fisher-z Test P-values.pdf",
    figsize=(30*CM, 8*CM),
)
