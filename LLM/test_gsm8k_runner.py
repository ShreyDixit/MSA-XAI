"""CPU checks; no model download or cluster allocation required."""
import contextlib
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from LLM import script as runner


class RunnerTests(unittest.TestCase):
    def test_atomic_json_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'config.json'
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda i: runner.atomic_json(target, {'writer': i}), range(64)))
            self.assertIn(json.loads(target.read_text())['writer'], range(64))
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_random_removals_are_reproducible_and_nested(self):
        population = tuple(product(range(32), range(8)))
        select = runner.removed_experts
        small = select(population, 5, 'random', 2810, 3)
        large = select(population, 70, 'random', 2810, 3)
        self.assertEqual(small, select(population, 5, 'random', 2810, 3))
        self.assertEqual(len(set(large)), 70)
        self.assertEqual(small, large[:5])
        self.assertNotEqual(small, select(population, 5, 'random', 2810, 4))

    def test_ranking_aligns_metric_names_and_shuffled_rows(self):
        ranking = runner.rank_experts([
            runner.SCRIPT_DIR / 'shapley_analysis_20240614-002232.csv',
            runner.SCRIPT_DIR / 'shapley_analysis_20240805-062942.csv'])
        self.assertEqual(set(ranking), set(product(range(32), range(8))))
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for index in range(2):
                path = Path(tmp) / f'{index}.csv'
                with path.open('w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Contribution', 'Performance Metric', 'Layer', 'Expert'])
                    rows = [(expert, metric + ('_accuracy' if index else ''), 0, expert)
                            for expert in range(2)
                            for metric in ['addition', 'subtraction', 'multiplication']]
                    writer.writerows(rows[::-1] if index else rows)
                paths.append(path)
            self.assertEqual(runner.rank_experts(paths), ((0, 0), (0, 1)))

    def test_tiny_model_lesion_and_dispatch_boundaries(self):
        from transformers import MixtralConfig
        cls = runner.model_class()
        config = MixtralConfig(hidden_size=16, intermediate_size=24,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            num_local_experts=2, num_experts_per_tok=2, vocab_size=32)
        model = cls(config).eval()
        self.assertIn('LesionedMixtralDecoderLayer', model._get_no_split_modules('auto'))
        block = model.model.layers[0].block_sparse_moe
        x = torch.randn(1, 2, 16)
        runner.lesion_dictionary.clear()
        intact, _ = block(x)
        runner.lesion_dictionary[0] = [0, 1]
        removed, _ = block(x)
        self.assertGreater(intact.abs().sum().item(), 0)
        self.assertEqual(removed.abs().sum().item(), 0)
        runner.lesion_dictionary.clear()

    def test_seeded_dataset_does_not_depend_on_global_random_state(self):
        import sys
        with patch.dict(sys.modules):
            sys.path.insert(0, str(runner.SCRIPT_DIR))
            try:
                from GSM8K.dataset import GSM8KDataset
                tokenizer = SimpleNamespace(apply_chat_template=lambda m, tokenize: repr(m))
                first = GSM8KDataset(tokenizer, num_shots=3, num_examples=3, seed=42)
                random.seed(100)
                second = GSM8KDataset(tokenizer, num_shots=3, num_examples=3, seed=42)
                self.assertEqual([first[i] for i in range(3)], [second[i] for i in range(3)])
            finally:
                sys.path.pop(0)

    def test_batch_checkpoint_resume_and_export(self):
        import sys
        import transformers
        class Encoding(dict):
            def __getattr__(self, key): return self[key]
            def to(self, device): return self
        class Tokenizer:
            eos_token = '</s>'
            eos_token_id = 0
            def apply_chat_template(self, message, tokenize=False): return repr(message)
            def __call__(self, prompts, **kwargs):
                return Encoding(input_ids=torch.ones((len(prompts), 2), dtype=torch.long),
                                attention_mask=torch.ones((len(prompts), 2), dtype=torch.long))
            def batch_decode(self, tokens, **kwargs): return ['#### 42'] * len(tokens)
        calls = []
        stop_after_batch = [True]
        class Model:
            config = SimpleNamespace(num_hidden_layers=32, num_local_experts=8)
            hf_device_map = {'': 'cpu'}
            def eval(self): return self
            def get_input_embeddings(self):
                return SimpleNamespace(weight=torch.zeros(1))
            def generate(self, **kwargs):
                calls.append(kwargs)
                if stop_after_batch[0]: runner.STOP = True
                inputs = kwargs['input_ids']
                return torch.ones((len(inputs), inputs.shape[1] + 1), dtype=torch.long)
        with tempfile.TemporaryDirectory() as tmp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
            stack.enter_context(patch.object(torch.cuda, 'device_count', return_value=2))
            stack.enter_context(patch.object(transformers.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()))
            stack.enter_context(patch.object(transformers, 'BitsAndBytesConfig'))
            stack.enter_context(patch.object(transformers, 'set_seed'))
            loader = stack.enter_context(patch.object(runner, 'model_class', return_value=SimpleNamespace(from_pretrained=lambda *a, **k: Model())))
            stack.enter_context(patch.object(runner.signal, 'signal'))
            stack.enter_context(patch.object(sys, 'path', [str(runner.SCRIPT_DIR), *sys.path]))
            argv = ['script.py', '--output-dir', tmp, '--conditions', 'random', '--counts', '5',
                    '--num-trials', '1', '--num-examples', '3', '--batch-size', '2']
            stack.enter_context(patch.object(sys, 'argv', argv))
            runner.STOP = False
            runner.main()
            path = runner.task_path(Path(tmp), 'random', 5, 0)
            partial = json.loads(path.read_text())
            self.assertEqual(partial['total'], 2)
            self.assertFalse(partial['complete'])
            runner.STOP = False
            stop_after_batch[0] = False
            runner.main()
            complete = json.loads(path.read_text())
            self.assertEqual(complete['total'], 3)
            self.assertTrue(complete['complete'])
            self.assertEqual(len(calls), 2)
            runner.main()
            self.assertEqual(loader.call_count, 2)  # Completed work never loads a model.
            args = runner.parse_args()
            runner.merge_results(args)
            self.assertEqual(json.loads((Path(tmp) / 'trial_ids_random.json').read_text()), {'5': [0]})
            self.assertEqual(len(json.loads((Path(tmp) / 'accuracies_random.json').read_text())['5']), 1)
            sys.argv[-1] = '1'  # Changing batch size must reject mixed results.
            with self.assertRaisesRegex(ValueError, 'different settings'):
                runner.main()
        runner.STOP = False


if __name__ == '__main__':
    unittest.main()
