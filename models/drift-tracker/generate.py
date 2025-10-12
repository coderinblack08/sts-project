import json
import random
import re
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from datasets import load_dataset

dataset_dir = Path(__file__).parent / "dataset"


def load_squad_dataset(split: Literal['train', 'test']) -> List[dict]:
    if split == 'train':
        with open(dataset_dir / "SQuAD-v2.0_train.json", "r") as f:
            squad_train = json.load(f)
        return squad_train
    elif split == 'test':
        with open(dataset_dir / "SQuAD-v2.0_dev.json", "r") as f:
            squad_dev = json.load(f)
        return squad_dev


def load_alpaca_dataset(split: Literal['train', 'test']):
    alpaca = load_dataset("tatsu-lab/alpaca")
    return alpaca[split]


def load_trigger_dataset(split: Literal['train', 'test']) -> List[str]:
    with open(dataset_dir / f"injection_trigger_{split}.txt", "r") as f:
        triggers = [line.strip() for line in f.readlines() if line.strip()]
    return triggers


def format_squad_qa(paragraph: dict, qas: List[dict]) -> List[Tuple[str, str]]:
    formatted_qa = []
    context = paragraph['context']
    for qa in qas:
        question = qa['question']
        formatted_qa.append((question, context))
    return formatted_qa


def format_alpaca_instruction(item: dict) -> str:
    instruction = item['instruction']
    return instruction


def inject_text(
    primary: Tuple[str, str],
    trigger: str,
    secondary: str,
    position: Literal['beginning', 'middle', 'end'],
) -> Tuple[str, int, int]:

    injection = trigger + " " + secondary

    if position == 'beginning':
        prefix = primary[0] + " "
        suffix = " " + primary[1]
        start_idx = len(prefix)
        end_idx = start_idx + len(injection)
        return prefix + injection + suffix, start_idx, end_idx
    elif position == 'end':
        prefix = primary[0] + " " + primary[1] + " "
        start_idx = len(prefix)
        end_idx = start_idx + len(injection)
        return prefix + injection, start_idx, end_idx
    else:
        sentences = re.split(r'(?<=[.!?])\s+', primary[1])
        if len(sentences) > 2:
            split_idx = random.randint(1, len(sentences) - 1)
        else:
            split_idx = 1
        first_half = ' '.join(sentences[:split_idx])
        second_half = ' '.join(sentences[split_idx:])
        prefix = primary[0] + " " + first_half + " "
        suffix = " " + second_half
        start_idx = len(prefix)
        end_idx = start_idx + len(injection)
        return prefix + injection + suffix, start_idx, end_idx


def generate_injections(split: Literal['train', 'test']) -> List[Dict[str, any]]:
    if split == 'train':
        squad_data = load_squad_dataset(split)
        alpaca_data = load_alpaca_dataset(split)
        triggers = load_trigger_dataset(split)

        alpaca_no_input = [
            item for item in alpaca_data if not item.get('input', '').strip()
        ]

        squad_instances = []
        for article in squad_data['data']:
            for paragraph in article['paragraphs']:
                qas_formatted = format_squad_qa(paragraph, paragraph['qas'])
                squad_instances.extend(qas_formatted)

        print(f"Loaded {len(squad_instances)} SQuAD instances")
        print(f"Loaded {len(alpaca_no_input)} Alpaca instances (no input)")
        print(f"Loaded {len(triggers)} triggers")

        samples = []
        positions = ['beginning', 'middle', 'end']

        num_samples = len(squad_instances)

        for i in range(num_samples):
            squad_idx = i % len(squad_instances)
            alpaca_idx = random.randint(0, len(alpaca_no_input) - 1)
            trigger_idx = random.randint(0, len(triggers) - 1)
            position = positions[i % 3]

            primary_text = squad_instances[squad_idx]
            secondary_text = format_alpaca_instruction(alpaca_no_input[alpaca_idx])
            trigger = triggers[trigger_idx]

            clean_sample = primary_text[0] + " " + primary_text[1]
            poisoned_sample, start_idx, end_idx = inject_text(
                primary_text, trigger, secondary_text, position
            )

            samples.append(
                {
                    "clean": clean_sample,
                    "poisoned": poisoned_sample,
                    "injection_start": start_idx,
                    "injection_end": end_idx,
                    "position": position,
                }
            )

        print(f"Generated {len(samples)} clean/poisoned sample pairs")
        return samples
    else:
        pass


if __name__ == "__main__":
    samples = generate_injections('train')

    with open(dataset_dir / f"injection_train.json", "w") as f:
        json.dump(samples, f, indent=2)
