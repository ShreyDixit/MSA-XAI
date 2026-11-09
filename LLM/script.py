# %%
import json
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
)

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from Levenshtein import ratio
from msapy import msa
import pickle as pkl
import time
from itertools import product
from functools import cache

from transformers.models.mixtral.modeling_mixtral import *
from transformers.models.mixtral import *
from tasks import *
from fastprogress import progress_bar, master_bar

from GSM8K import dataset

import warnings
warnings.filterwarnings( "ignore" )

# %%
SEED = 2810

# %%
# Initial Setup
num_permutations = 300 # Set this to 1000 to perform the full analysis. This takes a while to run. For the demonstration purpose, we set it to 1.
lesion_dictionary = defaultdict(list) # Initialize the lesion dictionary as empty meaning the network is intact
with torch.no_grad():
    torch.cuda.empty_cache()

# %%
device_map = "cuda:0"
model_path = "mistralai/Mixtral-8x7B-Instruct-v0.1"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# %%
num_arithmetic_operations_questions_train = 30
num_arithmetic_operations_questions_test = 50
total_arithmetic_operations_questions = num_arithmetic_operations_questions_train + num_arithmetic_operations_questions_test
num_arithmetic_operators = 3

num_sentences_per_language = 15
num_languages = 5

# %%
class LesionedMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ## Most of the code is copied from the original implementation
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            # if the current expert is lesioned, skip it. This is the only change I made
            if expert_idx in lesion_dictionary[self.layer_idx]:
                continue 

            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

class LesionedMixtralDecoderLayer(MixtralDecoderLayer):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.block_sparse_moe = LesionedMixtralSparseMoeBlock(config, layer_idx)

class LesionedMixtralModel(MixtralModel):
    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self._no_split_modules = ["LesionedMixtralDecoderLayer"]
        self.layers = nn.ModuleList(
            [LesionedMixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

class LesionedMixtralForCausalLM(MixtralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LesionedMixtralModel(config)
        # Initialize weights and apply final processing
        self.post_init()


# %%
lesioned_model = LesionedMixtralForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        quantization_config=quantization_config,
)

# %%
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# %%
rng = np.random.default_rng(SEED)

# %%

datasetGSM8k = dataset.GSM8KDataset(tokenizer, num_shots=3, num_examples=-1)
dataloaderGSM8k = DataLoader(datasetGSM8k, batch_size=64, shuffle=True)


# %%

def complements_to_lesion_dict(coplements):
    global lesion_dictionary
    lesion_dictionary = defaultdict(list)
    for c in coplements:
        lesion_dictionary[c[0]].append(c[1])

def evaluate_on_gsm8k(dataloaderGSM8k, complements):
    complements_to_lesion_dict(complements)
    correct_answers = 0
    total_questions = 0
    for X, y in dataloaderGSM8k:
        X = tokenizer(X, return_tensors="pt", padding=True).to(device_map)
        outputs = lesioned_model.generate(**X, max_new_tokens=350, pad_token_id = tokenizer.eos_token_id)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        answers = [dataset.extract_answer(output.split("INST")[-1]) for output in outputs]
        correct_answers += sum(1 for i in range(len(answers)) if answers[i] == y[i])
        total_questions += len(answers)

    return correct_answers / total_questions

# %%
shapley_df_formatted = pd.read_csv("shapley_analysis_20240614-002232.csv")
shapley_df_formatted["Contribution"] = ((pd.read_csv("shapley_analysis_20240805-062942.csv")["Contribution"] * 3) + (shapley_df_formatted["Contribution"] * 4)) / 7
arithmetic_tasks = ["addition", "subtraction", "multiplication"]
arithmetic_shapley_df = shapley_df_formatted[shapley_df_formatted["Performance Metric"].isin(arithmetic_tasks)]
average_arithmetic_shapley_df = arithmetic_shapley_df.groupby(["Layer", "Expert"])["Contribution"].mean().reset_index()
average_contrib_sorted = average_arithmetic_shapley_df.sort_values(by="Contribution", ascending=True)
low_contrib_experts = tuple(zip(average_contrib_sorted["Layer"], average_contrib_sorted["Expert"]))
high_contrib_experts = low_contrib_experts[::-1]

num_trials = 20
lesioning_experts_array = [1, *list(range(0, 75, 5))]

def gsm8k_experiment(experts, num_trials, lesioning_experts_array, suffix):
    accuracies = defaultdict(list)
    time_taken = defaultdict(list)

    try:
        with open(f"accuracies_{suffix}.json", "r") as f:
            accuracies = defaultdict(list, json.load(f)) # json.load(f)

        with open(f"time_taken_{suffix}.json", "r") as f:
            time_taken = defaultdict(list, json.load(f))

    except FileNotFoundError:
        pass

    for num_experts in lesioning_experts_array:
        print(f"num_experts: {num_experts}")

        if num_experts in accuracies or str(num_experts) in accuracies:
            print(f"Skipping {num_experts}")
            continue

        for i in progress_bar(range(num_trials)):
            start_time = time.time()
            accuracies[num_experts].append(evaluate_on_gsm8k(dataloaderGSM8k, experts[:num_experts]))
            end_time = time.time()

            time_taken[num_experts].append(end_time - start_time)
            with open(f"accuracies_{suffix}.json", "w") as f:
                json.dump(accuracies, f)

            with open(f"time_taken_{suffix}.json", "w") as f:
                json.dump(time_taken, f)

gsm8k_experiment(low_contrib_experts, num_trials, lesioning_experts_array, "low")
print("low done")
gsm8k_experiment(high_contrib_experts, num_trials, lesioning_experts_array, "high")
print("high done")


