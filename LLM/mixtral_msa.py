from __future__ import annotations

import argparse
import ast
import time
import warnings
from collections import defaultdict
from functools import cache
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from msapy import msa
from transformers import AutoTokenizer, BitsAndBytesConfig
from transformers.models.mixtral.modeling_mixtral import (
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralForCausalLM,
    MixtralModel,
    MixtralSparseMoeBlock,
)

from tasks import (
    extract_capital_from_output,
    extract_language_from_output,
    get_addition_question,
    get_arithmetic_accuracy,
    get_arithmetic_messages,
    get_countries_capitals_messages,
    get_language_sentence_message,
    get_multiplication_question,
    get_subtraction_question,
)

warnings.filterwarnings("ignore")

SEED = 2810
MODEL_PATH = "mistralai/Mixtral-8x7B-Instruct-v0.1"
SCRIPT_DIR = Path(__file__).resolve().parent

NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TRAIN = 30
NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TEST = 50
TOTAL_ARITHMETIC_OPERATIONS_QUESTIONS = (
    NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TRAIN
    + NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TEST
)
NUM_SENTENCES_PER_LANGUAGE = 15
ARITHMETIC_METRICS = [
    "addition_accuracy",
    "subtraction_accuracy",
    "multiplication_accuracy",
]
LANGUAGE_METRICS = [
    "english_accuracy",
    "german_accuracy",
    "french_accuracy",
    "spanish_accuracy",
    "italian_accuracy",
]
FACTUAL_KNOWLEDGE_METRICS = [
    "country_capital_accuracy",
]

lesion_dictionary: dict[int, list[int]] = defaultdict(list)


class LesionedMixtralSparseMoeBlock(MixtralSparseMoeBlock):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__(config)
        self.layer_idx = layer_idx

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise,
                1.0 + self.jitter_noise,
            )

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights,
            self.top_k,
            dim=-1,
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        expert_mask = torch.nn.functional.one_hot(
            selected_experts,
            num_classes=self.num_experts,
        ).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            if expert_idx in lesion_dictionary[self.layer_idx]:
                continue

            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(
                0,
                top_x,
                current_hidden_states.to(hidden_states.dtype),
            )

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

    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.model = LesionedMixtralModel(config)
        self.post_init()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Mixtral MSA up to the first msa.interface(...) call.",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=100,
        help="Number of permutations passed to msapy.msa.interface.",
    )
    parser.add_argument(
        "--save-permutations",
        action="store_true",
        help=(
            "Save per-permutation marginal contributions and cumulative means "
            "for convergence plots."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for data generation, model sampling, and MSA permutations.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Load the model in 4-bit quantized mode.",
    )
    parser.add_argument(
        "--arithmetic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the arithmetic task group.",
    )
    parser.add_argument(
        "--language",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable the language task group.",
    )
    parser.add_argument(
        "--factual-knowledge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the factual knowledge task group.",
    )
    parser.add_argument(
        "--model-path",
        default=MODEL_PATH,
        help="Hugging Face model identifier or local checkpoint path.",
    )
    return parser.parse_args()


def resolve_compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model(model_path: str, quantize: bool, compute_dtype: torch.dtype) -> LesionedMixtralForCausalLM:
    model_kwargs: dict[str, object] = {
        "device_map": "auto",
    }

    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    return LesionedMixtralForCausalLM.from_pretrained(model_path, **model_kwargs)


def complements_to_lesion_dict(complements: tuple[tuple[int, int], ...]) -> None:
    global lesion_dictionary
    lesion_dictionary = defaultdict(list)
    for layer_idx, expert_idx in complements:
        lesion_dictionary[layer_idx].append(expert_idx)


def get_active_metric_names(
    arithmetic: bool,
    language: bool,
    factual_knowledge: bool,
) -> list[str]:
    active_metric_names: list[str] = []
    if arithmetic:
        active_metric_names.extend(ARITHMETIC_METRICS)
    if language:
        active_metric_names.extend(LANGUAGE_METRICS)
    if factual_knowledge:
        active_metric_names.extend(FACTUAL_KNOWLEDGE_METRICS)
    return active_metric_names


def build_task_descriptor(
    arithmetic: bool,
    language: bool,
    factual_knowledge: bool,
) -> str:
    return "_".join(
        [
            f"arithmetic_{'on' if arithmetic else 'off'}",
            f"language_{'on' if language else 'off'}",
            f"factual_knowledge_{'on' if factual_knowledge else 'off'}",
        ]
    )


def build_output_path(
    quantize: bool,
    num_permutations: int,
    save_permutations: bool,
    arithmetic: bool,
    language: bool,
    factual_knowledge: bool,
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    task_descriptor = build_task_descriptor(arithmetic, language, factual_knowledge)
    return SCRIPT_DIR / (
        f"shapley_analysis_{task_descriptor}"
        f"_quantized_{str(quantize).lower()}"
        f"_save_permutations_{str(save_permutations).lower()}"
        f"_num_permutations_{num_permutations}_{timestamp}.csv"
    )


def format_shapley_table(
    shapley_table: pd.DataFrame,
    active_metric_names: list[str],
) -> pd.DataFrame:
    shapley_df = pd.DataFrame(shapley_table).copy()
    shapley_df.columns = shapley_df.columns.astype(str)
    shapley_df_formatted = shapley_df.melt(
        ignore_index=False,
        var_name="Layer_Expert",
        value_name="Contribution",
    )
    task_mapping = dict(enumerate(active_metric_names))
    shapley_df_formatted["Performance Metric"] = shapley_df_formatted.index.map(task_mapping)
    layer_expert_pairs = shapley_df_formatted["Layer_Expert"].apply(ast.literal_eval)
    shapley_df_formatted[["Layer", "Expert"]] = pd.DataFrame(
        layer_expert_pairs.tolist(),
        index=shapley_df_formatted.index,
    )
    shapley_df_formatted.drop(columns=["Layer_Expert"], inplace=True)
    return shapley_df_formatted


def format_permutation_contributions(
    shapley_table: pd.DataFrame,
    active_metric_names: list[str],
) -> pd.DataFrame:
    permutation_df = pd.DataFrame(shapley_table).copy()
    permutation_df.columns = permutation_df.columns.astype(str)
    permutation_df.index = permutation_df.index.set_names(["Permutation", "Metric_Index"])

    permutation_df_formatted = permutation_df.reset_index().melt(
        id_vars=["Permutation", "Metric_Index"],
        var_name="Layer_Expert",
        value_name="Contribution",
    )
    permutation_df_formatted["Performance Metric"] = permutation_df_formatted[
        "Metric_Index"
    ].map(dict(enumerate(active_metric_names)))
    layer_expert_pairs = permutation_df_formatted["Layer_Expert"].apply(ast.literal_eval)
    permutation_df_formatted[["Layer", "Expert"]] = pd.DataFrame(
        layer_expert_pairs.tolist(),
        index=permutation_df_formatted.index,
    )
    permutation_df_formatted.sort_values(
        ["Performance Metric", "Layer", "Expert", "Permutation"],
        inplace=True,
    )
    permutation_df_formatted["Num Permutations"] = (
        permutation_df_formatted.groupby(["Metric_Index", "Layer", "Expert"]).cumcount() + 1
    )
    permutation_df_formatted["Running Contribution"] = permutation_df_formatted.groupby(
        ["Metric_Index", "Layer", "Expert"]
    )["Contribution"].transform(lambda values: values.expanding().mean())
    permutation_df_formatted.drop(columns=["Layer_Expert"], inplace=True)
    return permutation_df_formatted


def main() -> None:
    args = parse_args()
    if not any([args.arithmetic, args.language, args.factual_knowledge]):
        raise ValueError("At least one task group must be enabled.")

    compute_dtype = resolve_compute_dtype()
    device = "cuda:0"
    active_metric_names = get_active_metric_names(
        arithmetic=args.arithmetic,
        language=args.language,
        factual_knowledge=args.factual_knowledge,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with torch.no_grad():
        torch.cuda.empty_cache()

    lesioned_model = build_model(
        model_path=args.model_path,
        quantize=args.quantize,
        compute_dtype=compute_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if args.arithmetic:
        rng = np.random.default_rng(args.seed)
        (
            addition_messages_train,
            addition_answers_train,
            addition_messages_test,
            addition_answers_test,
        ) = get_arithmetic_messages(
            *get_addition_question(rng, size=TOTAL_ARITHMETIC_OPERATIONS_QUESTIONS),
            "+",
            num_training_examples=NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TRAIN,
        )
        (
            subtraction_messages_train,
            subtraction_answers_train,
            subtraction_messages_test,
            subtraction_answers_test,
        ) = get_arithmetic_messages(
            *get_subtraction_question(rng, size=TOTAL_ARITHMETIC_OPERATIONS_QUESTIONS),
            "-",
            num_training_examples=NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TRAIN,
        )
        (
            multiplication_messages_train,
            multiplication_answers_train,
            multiplication_messages_test,
            multiplication_answers_test,
        ) = get_arithmetic_messages(
            *get_multiplication_question(rng, size=TOTAL_ARITHMETIC_OPERATIONS_QUESTIONS),
            "*",
            num_training_examples=NUM_ARITHMETIC_OPERATIONS_QUESTIONS_TRAIN,
        )

        all_arithmetic_messages_train = (
            addition_messages_train + subtraction_messages_train + multiplication_messages_train
        )
        all_arithmetic_messages_test = (
            addition_messages_test + subtraction_messages_test + multiplication_messages_test
        )

        all_arithmetic_answers_train = np.concatenate(
            [addition_answers_train, subtraction_answers_train, multiplication_answers_train]
        )
        all_arithmetic_answers_test = np.concatenate(
            [addition_answers_test, subtraction_answers_test, multiplication_answers_test]
        )

        all_arithmetic_messages_train = [
            tokenizer.apply_chat_template(message, tokenize=False)
            for message in all_arithmetic_messages_train
        ]
        all_arithmetic_messages_test = [
            tokenizer.apply_chat_template(message, tokenize=False)
            for message in all_arithmetic_messages_test
        ]

        arithmetic_inputs_train = tokenizer(
            all_arithmetic_messages_train,
            return_tensors="pt",
            padding=True,
        ).to(device)
        arithmetic_inputs_test = tokenizer(
            all_arithmetic_messages_test,
            return_tensors="pt",
            padding=True,
        ).to(device)

        def get_operation_wise_accuracy(test: bool = False) -> list[float]:
            answers = all_arithmetic_answers_test if test else all_arithmetic_answers_train
            inputs = arithmetic_inputs_test if test else arithmetic_inputs_train
            outputs = lesioned_model.generate(
                **inputs,
                max_new_tokens=9,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return list(
                np.array(
                    [
                        get_arithmetic_accuracy(output, answer)
                        for output, answer in zip(outputs, answers)
                    ]
                ).reshape(3, -1).mean(axis=1)
            )

    if args.language:
        languaged_sentences_data = pd.read_csv(SCRIPT_DIR / "sentences.csv")
        language_sentences = get_language_sentence_message(
            "eng",
            languaged_sentences_data,
            num_sentences=NUM_SENTENCES_PER_LANGUAGE,
        )
        language_sentences += get_language_sentence_message(
            "deu",
            languaged_sentences_data,
            num_sentences=NUM_SENTENCES_PER_LANGUAGE,
        )
        language_sentences += get_language_sentence_message(
            "fra",
            languaged_sentences_data,
            num_sentences=NUM_SENTENCES_PER_LANGUAGE,
        )
        language_sentences += get_language_sentence_message(
            "spa",
            languaged_sentences_data,
            num_sentences=NUM_SENTENCES_PER_LANGUAGE,
        )
        language_sentences += get_language_sentence_message(
            "ita",
            languaged_sentences_data,
            num_sentences=NUM_SENTENCES_PER_LANGUAGE,
        )

        language_sentences = [
            tokenizer.apply_chat_template(message, tokenize=False)
            for message in language_sentences
        ]
        language_inputs = tokenizer(language_sentences, return_tensors="pt", padding=True).to(device)
        languages = ["english", "deutsch", "français", "español", "italiano"]

        def get_language_wise_accuracy() -> list[float]:
            outputs = lesioned_model.generate(
                **language_inputs,
                max_new_tokens=3,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs = [extract_language_from_output(output) for output in outputs]

            language_wise_accuracy: list[float] = []
            for i, language in enumerate(languages):
                outputs_language = outputs[
                    i * NUM_SENTENCES_PER_LANGUAGE : (i + 1) * NUM_SENTENCES_PER_LANGUAGE
                ]
                num_right_answers_language = len(
                    [1 for output in outputs_language if language in output]
                )
                language_wise_accuracy.append(num_right_answers_language / NUM_SENTENCES_PER_LANGUAGE)
            return language_wise_accuracy

    if args.factual_knowledge:
        countries_capitals_data = pd.read_csv(SCRIPT_DIR / "country_capitals.csv")
        countries_capitals_data.capital = countries_capitals_data.capital.str.lower()

        country_capital_messages = get_countries_capitals_messages(countries_capitals_data)
        country_capital_messages = [
            tokenizer.apply_chat_template(message, tokenize=False)
            for message in country_capital_messages
        ]
        country_capital_inputs = tokenizer(
            country_capital_messages,
            return_tensors="pt",
            padding=True,
        ).to(device)
        capitals_answers = countries_capitals_data["capital"].values

        def get_country_capital_accuracy() -> list[float]:
            outputs = lesioned_model.generate(
                **country_capital_inputs,
                max_new_tokens=3,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs = [extract_capital_from_output(output) for output in outputs]
            correct_answers = len(
                [1 for output, capital in zip(outputs, capitals_answers) if capital in output]
            )
            return [correct_answers / len(outputs)]

    @cache
    @torch.no_grad()
    def objective_function_cached(
        complements: tuple[tuple[int, int], ...],
        test: bool = False,
    ) -> np.ndarray:
        complements_to_lesion_dict(complements)
        metrics: list[float] = []
        if args.arithmetic:
            metrics.extend(get_operation_wise_accuracy(test=test))
        if args.language:
            metrics.extend(get_language_wise_accuracy())
        if args.factual_knowledge:
            metrics.extend(get_country_capital_accuracy())
        return np.array(metrics)

    def objective_function(
        complements: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        test: bool = False,
    ) -> np.ndarray:
        normalized_complements = tuple(tuple(item) for item in complements)
        return objective_function_cached(normalized_complements, test=test)

    unlesioned_model_performance = objective_function(())
    print("Unlesioned model performance:")
    for metric_name, metric_value in zip(active_metric_names, unlesioned_model_performance.tolist()):
        print(f"  {metric_name}={metric_value:.4f}")

    elements = list(product(range(32), range(8)))

    shapley_table = msa.interface(
        n_permutations=args.num_permutations,
        elements=elements,
        objective_function=objective_function,
        #random_seed=args.seed,
        save_permutations=args.save_permutations,
    )

    if args.save_permutations:
        permutation_df_formatted = format_permutation_contributions(
            shapley_table,
            active_metric_names,
        )
        shapley_table = pd.DataFrame(shapley_table).copy()
        shapley_table.index = shapley_table.index.set_names(["Permutation", "Metric_Index"])
        shapley_table = shapley_table.groupby(level="Metric_Index").mean()

    shapley_df_formatted = format_shapley_table(shapley_table, active_metric_names)

    output_path = build_output_path(
        quantize=args.quantize,
        num_permutations=args.num_permutations,
        save_permutations=args.save_permutations,
        arithmetic=args.arithmetic,
        language=args.language,
        factual_knowledge=args.factual_knowledge,
    )
    shapley_df_formatted.to_csv(output_path, index=False)

    if args.save_permutations:
        permutation_output_path = output_path.with_name(
            f"{output_path.stem}_permutation_contributions.csv"
        )
        permutation_df_formatted.to_csv(permutation_output_path, index=False)
        print(f"Saved permutation contributions to {permutation_output_path}")

    print(f"Saved averaged results to {output_path}")
    print(
        "Run configuration:"
        f" quantized={args.quantize},"
        f" compute_dtype={compute_dtype},"
        f" num_permutations={args.num_permutations},"
        f" save_permutations={args.save_permutations},"
        f" seed={args.seed},"
        f" arithmetic={args.arithmetic},"
        f" language={args.language},"
        f" factual_knowledge={args.factual_knowledge}"
    )


if __name__ == "__main__":
    main()
