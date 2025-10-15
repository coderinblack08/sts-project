import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from torch import Tensor
from typing import Dict, List, Optional
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


def load_samples(n: Optional[int], split: str):
    with open(dataset_dir / f"injection_{split}_min.json", "r") as f:
        data = json.load(f)
    assert len(data) % 3 == 0
    sampled = [data[i + random.randint(0, 2)] for i in range(0, len(data), 3)]

    print(len(sampled), len(data))
    return sampled if n is None else sampled[:n]


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
    parser.add_argument(
        "--n",
        type=str,
        default="1",
        help="Number of samples to process (or 'max' for all)",
    )
    parser.add_argument(
        "--suffix", action="store_true", help="Do you want to add a suffix to the text?"
    )
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    args = parser.parse_args()

    n = None if args.n.lower() == "max" else int(args.n)

    print("Loading model...")
    model_name = "Qwen/Qwen2.5-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    print(f"Loading {args.n} samples...")
    samples = load_samples(n, args.split)

    all_results = []

    for idx, sample in tqdm(enumerate(samples), total=len(samples)):
        suffix = ""
        if args.suffix:
            suffix = "You were given a task above, but first before you answer, please complete the following sentence by briefly writing each request(s) you received and you are going to execute next: \"All requests that I am going to execute now are:\""

        clean_text = sample["clean"] + suffix
        poisoned_text = sample["poisoned"] + suffix
        clean_tokens = tokenizer(clean_text, return_tensors="pt")
        poisoned_tokens = tokenizer(poisoned_text, return_tensors="pt")

        end_of_clean_token = clean_tokens["input_ids"].shape[1] - 1
        end_of_poisoned_token = poisoned_tokens["input_ids"].shape[1] - 1
        end_of_primary_token = char_to_token_index(
            clean_text, sample["injection_start"] - 2, tokenizer
        )
        end_of_injection_token = char_to_token_index(
            poisoned_text, sample["injection_end"], tokenizer
        )

        clean_activations = get_activations_at_positions(
            model, tokenizer, clean_text, [end_of_primary_token, end_of_clean_token]
        )
        poisoned_activations = get_activations_at_positions(
            model,
            tokenizer,
            poisoned_text,
            [end_of_primary_token, end_of_injection_token, end_of_poisoned_token],
        )

        all_results.append(
            {
                "clean": clean_activations,
                "poisoned": poisoned_activations,
                "metadata": {
                    "sample_idx": idx,
                    "end_of_primary_token": end_of_primary_token,
                    "end_of_injection_token": end_of_injection_token,
                    "end_of_clean_token": end_of_clean_token,
                    "end_of_poisoned_token": end_of_poisoned_token,
                },
            }
        )

    output_data = {
        "samples": all_results,
        "global_metadata": {
            "num_samples": len(samples),
            "num_layers": len(model.model.layers),
            "d_model": model.config.hidden_size,
        },
    }

    suffix_str = "_suffix" if args.suffix else ""
    torch.save(output_data, output_dir / f"activations{suffix_str}_{len(samples)}.pt")


if __name__ == "__main__":
    main()
