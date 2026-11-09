from copy import deepcopy
import json
import os
import random
import re
import torch as th


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("GSM8K/data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

class GSM8KDataset(th.utils.data.Dataset):
    def __init__(self, tokenizer, num_shots=1, num_examples = -1):
        dataset_train = get_examples("train")
        self.dataset_test = get_examples("test")        
        train_examples = random.sample(dataset_train, num_shots)

        if num_examples > 0:
            self.dataset_test = random.sample(self.dataset_test, num_examples)

        self.tokenizer = tokenizer

        message = [
            {
                "role": "user",
                "content": "You have to solve High School Maths Problems. The answer should be formatted such that the final numeric solution is the final line of the solution, preceded by ####"
            },
            {
                "role": "assistant",
                "content": "Okay"
            }
        ]
        for example in train_examples:
            message.append({
                "role": "user",
                "content": example["question"]
            })
            message.append({
                "role": "assistant",
                "content": example["answer"]
            })

        self.message = message

    def __len__(self):
        return len(self.dataset_test)

    def __getitem__(self, idx):
        test_example = self.dataset_test[idx]
        message = deepcopy(self.message)
        message.append({
            "role": "user",
            "content": test_example["question"]
        })

        answer = extract_answer(test_example["answer"])
        return self.tokenizer.apply_chat_template(message, tokenize=False), answer