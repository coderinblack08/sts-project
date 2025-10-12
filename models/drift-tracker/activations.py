import argparse
import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from torch import Tensor
from typing import Dict, List
import random
from tqdm import tqdm

dataset_dir = Path(__file__).parent / "dataset"
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_samples(n: int):
    with open(dataset_dir / "injection_train_min.json", "r") as f:
        data = json.load(f)
    sampled = [data[i + random.randint(0, 2)] for i in range(0, len(data), 3)]

    print(len(sampled), len(data))
    return sampled[:n]


def char_to_token_index(text: str, char_idx: int, tokenizer) -> int:
    tokens = tokenizer(text, return_offsets_mapping=True)
    offsets = tokens['offset_mapping']

    for token_idx, (start, end) in enumerate(offsets):
        if start <= char_idx < end:
            return token_idx
    return len(offsets) - 1


def get_activations_at_positions(
    model,
    tokenizer,
    text: str,
    positions: List[int],
) -> Dict[str, Float[Tensor, "n_layers d_model"]]:

    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    activations_by_position = {}

    for pos_idx in positions:
        layer_activations = []

        def hook_fn(layer_idx):
            def hook(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                activation = hidden_states[0, pos_idx, :].detach().cpu()
                layer_activations.append(activation)

            return hook

        hooks = []
        for layer_idx, layer in enumerate(model.model.layers):
            hook = layer.register_forward_hook(hook_fn(layer_idx))
            hooks.append(hook)

        with torch.no_grad():
            model(input_ids)

        for hook in hooks:
            hook.remove()

        activations_by_position[f"pos_{pos_idx}"] = torch.stack(layer_activations)

    return activations_by_position


def main():
    parser = argparse.ArgumentParser(description="Extract activations from Qwen model")
    parser.add_argument("--n", type=int, default=1, help="Number of samples to process")
    args = parser.parse_args()

    start_time = time.time()

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    print(f"Loading {args.n} samples...")
    samples = load_samples(args.n)

    all_results = []
    total_clean_time = 0
    total_poisoned_time = 0

    for idx, sample in tqdm(
        enumerate(samples), total=args.n, desc="Processing samples"
    ):
        print(f"\n[{idx+1}/{args.n}] Processing sample {idx}...")

        clean_start = time.time()
        clean_text = sample["clean"]
        clean_tokens = tokenizer(clean_text, return_tensors="pt")
        last_token_pos = clean_tokens["input_ids"].shape[1] - 1

        clean_activations = get_activations_at_positions(
            model, tokenizer, clean_text, [last_token_pos]
        )
        clean_time = time.time() - clean_start
        total_clean_time += clean_time

        poisoned_start = time.time()
        poisoned_text = sample["poisoned"]
        injection_start_char = sample["injection_start"]
        injection_end_char = sample["injection_end"]

        injection_start_token = char_to_token_index(
            poisoned_text, injection_start_char, tokenizer
        )
        injection_end_token = char_to_token_index(
            poisoned_text, injection_end_char, tokenizer
        )

        poisoned_tokens = tokenizer(poisoned_text, return_tensors="pt")
        last_token_pos_poisoned = poisoned_tokens["input_ids"].shape[1] - 1

        positions_to_save = [
            injection_start_token,
            injection_end_token,
            last_token_pos_poisoned,
        ]

        poisoned_activations = get_activations_at_positions(
            model, tokenizer, poisoned_text, positions_to_save
        )
        poisoned_time = time.time() - poisoned_start
        total_poisoned_time += poisoned_time

        all_results.append(
            {
                "clean": clean_activations,
                "poisoned": poisoned_activations,
                "metadata": {
                    "sample_idx": idx,
                    "injection_start_token": injection_start_token,
                    "injection_end_token": injection_end_token,
                    "clean_last_token": last_token_pos,
                    "poisoned_last_token": last_token_pos_poisoned,
                },
            }
        )

    output_data = {
        "samples": all_results,
        "global_metadata": {
            "num_samples": args.n,
            "num_layers": len(model.model.layers),
            "d_model": model.config.hidden_size,
        },
    }

    torch.save(output_data, output_dir / f"activations_{args.n}.pt")

    total_time = time.time() - start_time
    print(f"\nSaved all activations to activations_{args.n}.pt")
    print(f"Processed {args.n} samples")
    print(f"Avg clean time: {total_clean_time/args.n:.2f}s")
    print(f"Avg poisoned time: {total_poisoned_time/args.n:.2f}s")
    print(f"Total time: {total_time:.2f}s")


if __name__ == "__main__":
    main()
