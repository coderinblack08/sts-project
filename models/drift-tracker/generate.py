import json
import random
import re
import argparse
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


def load_secondary_tests_dataset() -> List[str]:
    advbench = load_dataset("walledai/AdvBench")
    advbench_prompts = [item['prompt'] for item in advbench['train']]

    jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    jbb_goals = [item['Goal'] for item in jbb['harmful']]

    beavertails = load_dataset("PKU-Alignment/BeaverTails")
    beavertails_prompts = [
        item['prompt'] for item in beavertails['30k_test'] if not item['is_safe']
    ]

    all_prompts = advbench_prompts + jbb_goals + beavertails_prompts
    print(f"Loaded {len(advbench_prompts)} AdvBench prompts")
    print(f"Loaded {len(jbb_goals)} JailbreakBench goals")
    print(f"Loaded {len(beavertails_prompts)} BeaverTails unsafe prompts")
    print(f"Total secondary test prompts: {len(all_prompts)}")

    return all_prompts


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


def process_split_data(
    squad_data: dict,
    triggers: List[str],
    alpaca_data=None,
    secondary_data: List[str] = None,
) -> List[Dict[str, any]]:
    if secondary_data is not None:
        secondary_instructions = secondary_data
        print(f"Using {len(secondary_instructions)} secondary test prompts")
    else:
        alpaca_no_input = [
            item for item in alpaca_data if not item.get('input', '').strip()
        ]
        secondary_instructions = alpaca_no_input
        print(f"Loaded {len(alpaca_no_input)} Alpaca instances (no input)")

    squad_instances = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            qas_formatted = format_squad_qa(paragraph, paragraph['qas'])
            squad_instances.extend(qas_formatted)

    print(f"Loaded {len(squad_instances)} SQuAD instances")
    print(f"Loaded {len(triggers)} triggers")

    samples = []
    positions = ['beginning', 'middle', 'end']
    num_samples = len(squad_instances)

    for i in range(num_samples):
        squad_idx = i % len(squad_instances)
        secondary_idx = random.randint(0, len(secondary_instructions) - 1)
        trigger_idx = random.randint(0, len(triggers) - 1)
        position = positions[i % 3]

        primary_text = squad_instances[squad_idx]

        if secondary_data is not None:
            secondary_text = secondary_instructions[secondary_idx]
        else:
            secondary_text = secondary_instructions[secondary_idx]['instruction']

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


def generate_injections(split: Literal['train', 'test']) -> List[Dict[str, any]]:
    squad_data = load_squad_dataset(split)
    triggers = load_trigger_dataset(split)

    if split == 'test':
        secondary_data = load_secondary_tests_dataset()
        return process_split_data(squad_data, triggers, secondary_data=secondary_data)
    else:
        alpaca_data = load_alpaca_dataset(split)
        return process_split_data(squad_data, triggers, alpaca_data=alpaca_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate injections")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    samples = generate_injections(args.split)

    with open(dataset_dir / f"injection_{args.split}.json", "w") as f:
        json.dump(samples, f, indent=2)
