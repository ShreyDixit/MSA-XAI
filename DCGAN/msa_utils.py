from collections import defaultdict
from typing import Literal
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from msapy import msa, datastructures
from netneurotools import cluster
import bct

SEED = 2810

@torch.no_grad()
def convert_to_image_grid(contributions, normalize=True) -> np.ndarray:
    image_grid = vutils.make_grid(torch.Tensor(contributions), padding=2, normalize=normalize)
    image_grid = tensor_to_image(image_grid)

    return image_grid

def tensor_to_image(image_grid):
    image_grid = image_grid.detach().cpu().numpy().transpose(1, 2, 0)
    return image_grid

def get_layer_contributions(layer: int, shapley_modes: datastructures.ShapleyModeND) -> list:
    layer_contrib = []

    for x, y in shapley_modes.columns:
        if x == layer:
            layer_contrib.append(shapley_modes.get_shapley_mode((x, y)))
    return layer_contrib

def get_correlation_matrix(shapley_modes: datastructures.ShapleyModeND, layer: int = -1) -> np.ndarray:
    shapley_modes = shapley_modes if layer < 0 else shapley_modes[[node for node in shapley_modes.columns if node[0] == layer]]
    correlation = np.corrcoef(shapley_modes.values, rowvar=False)
    correlation = np.nan_to_num(correlation)
    return correlation

def lesion_list_to_lesion_dict(complements):
    lesion_dict = defaultdict(list)
    for x, y in complements:
        lesion_dict[x].append(y)
    return lesion_dict

@torch.no_grad()
def objective_function(complements, generator, fixed_noise):
    lesion_dict = lesion_list_to_lesion_dict(complements)
    return generator(fixed_noise, lesion_dict).detach().cpu().numpy()

@torch.no_grad()
def get_lesioned_output(elements: np.ndarray, cluster_labels: np.ndarray, cluster_ids: list, model: nn.Module, fixed_noise: torch.Tensor, example_id: int | None = None) -> list:
    model.eval()

    if example_id is not None:
        fixed_noise = fixed_noise[example_id:example_id+1]
    
    lesioned_elements = get_elements_in_cluster(elements, cluster_labels, cluster_ids)

    lesioned_output = model(fixed_noise, lesion_dict=lesion_list_to_lesion_dict(lesioned_elements))
    unlesioned_output = model(fixed_noise)
    difference = unlesioned_output - lesioned_output

    if example_id is None:
        return [convert_to_image_grid(lesioned_output, False), convert_to_image_grid(unlesioned_output, False), convert_to_image_grid(difference)]
    
    return [tensor_to_image(lesioned_output), tensor_to_image(unlesioned_output), tensor_to_image(difference)]

def get_elements_in_cluster(elements, cluster_labels, cluster_ids):
    lesioned_indices = np.where(np.isin(cluster_labels, cluster_ids))[0]
    lesioned_elements = [elements[i] for i in lesioned_indices]
    return lesioned_elements

def cluster_modes(shapley_modes: datastructures.ShapleyModeND, clustering_function: callable, example_indices: list | None = None, num_examples: int = 32, layer_indices: list | None = None) -> np.ndarray:
    np.random.RandomState(SEED)
    if layer_indices is not None:
        shapley_modes = shapley_modes[[node for node in shapley_modes.columns if node[0] in layer_indices]]
    
    values = shapley_modes.values

    if example_indices is not None:
        num_elements = len(shapley_modes.columns)
        values = values.reshape(num_examples, -1, num_elements)[example_indices].reshape(-1, num_elements)

    return clustering_function(values.T)

def cluster_community_louvain(x: np.ndarray, neg_handle_strategy: Literal['zero', 'abs'] = 'zero') -> np.ndarray:
    np.random.RandomState(SEED)
    corelation_matrix = np.corrcoef(x)

    if neg_handle_strategy == 'zero':
        corelation_matrix[corelation_matrix < 0] = 0
    elif neg_handle_strategy == 'abs':
        corelation_matrix = np.abs(corelation_matrix)

    ci = [bct.community_louvain(corelation_matrix, gamma=3, seed=n)[0] for n in range(100)]
    consensus = cluster.find_consensus(np.column_stack(ci), seed=SEED)

    return consensus

    
def get_all_cluster_contributions(elements: np.ndarray, cluster_labels: np.ndarray, model: nn.Module, fixed_noise: torch.Tensor):
    cluster_ids = np.unique(cluster_labels)
    cluster_contributions = []
    for cluster_id in cluster_ids:
        cluster_contributions.append(get_lesioned_output(elements, cluster_labels, cluster_id, model, fixed_noise)[0])
    return cluster_contributions