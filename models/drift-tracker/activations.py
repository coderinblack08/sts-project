import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float
from torch import Tensor
from typing import Dict, List
from tqdm import tqdm
from huggingface_hub import upload_file, create_repo
from utils import load_samples, output_dir

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


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

        def hook_fn():
            def hook(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output
                activation = hidden_states[0, pos_idx, :].detach().cpu()
                layer_activations.append(activation)

            return hook

        hooks = []
        for layer in model.model.layers:
            hook = layer.register_forward_hook(hook_fn())
            hooks.append(hook)

        with torch.no_grad():
            model(input_ids)

        for hook in hooks:
            hook.remove()

        activations_by_position[f"pos_{pos_idx}"] = torch.stack(layer_activations)

    return activations_by_position


def main():
    parser = argparse.ArgumentParser(description="Extract activations from Qwen model")
    parser.add_argument("--n", type=str, default="max")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--suffix", action="store_true")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=None)
    args = parser.parse_args()

    if args.hf and not args.hf_repo:
        parser.error("--hf-repo is required when --hf is specified")

    if args.hf and "/" not in args.hf_repo:
        parser.error("--hf-repo must be in format 'username/repo-name'")

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

    suffix_text = (
        "You were given a task above, but first before you answer, please complete the following sentence by briefly writing each request(s) you received and you are going to execute next: \"All requests that I am going to execute now are:\""
        if args.suffix
        else ""
    )

    data = []

    for idx, sample in tqdm(samples, total=len(samples)):
        clean_text = sample["clean"] + suffix_text
        poisoned_text = sample["poisoned"] + suffix_text
        clean_tokens = tokenizer(clean_text, return_tensors="pt")
        poisoned_tokens = tokenizer(poisoned_text, return_tensors="pt")

        end_of_primary_token = char_to_token_index(
            clean_text, sample["primary_end"], tokenizer
        )
        end_of_injection_token = char_to_token_index(
            poisoned_text, sample["injection_end"], tokenizer
        )
        start_of_injection_token = char_to_token_index(
            poisoned_text, sample["injection_start"], tokenizer
        )
        end_of_clean_token = clean_tokens["input_ids"].shape[1] - 1
        end_of_poisoned_token = poisoned_tokens["input_ids"].shape[1] - 1

        clean_activations = get_activations_at_positions(
            model, tokenizer, clean_text, [end_of_primary_token, end_of_clean_token]
        )
        poisoned_activations = get_activations_at_positions(
            model,
            tokenizer,
            poisoned_text,
            [end_of_poisoned_token],
        )

        result = {
            "sample_idx": idx,
            "positions": {
                "end_of_primary": end_of_primary_token,
                "start_of_injection": start_of_injection_token,
                "end_of_injection": end_of_injection_token,
                "end_of_clean": end_of_clean_token,
                "end_of_poisoned": end_of_poisoned_token,
            },
            "activations": {
                "clean": clean_activations,
                "poisoned": poisoned_activations,
            },
        }

        data.append(result)

    output_data = {
        "samples": data,
        "global_metadata": {
            "num_samples": len(samples),
            "num_layers": len(model.model.layers),
            "d_model": model.config.hidden_size,
        },
    }

    filename = f"{args.split}_activations_{'max' if args.n.lower() == 'max' else args.n}{len(samples)}{'_suffix' if args.suffix else ''}.pt"
    output_path = output_dir / filename
    torch.save(output_data, output_path)
    print(f"Saved activations to {output_path}")

    if args.hf:
        create_repo(repo_id=args.hf_repo, repo_type="dataset", exist_ok=True)
        upload_file(
            path_or_fileobj=str(output_path),
            path_in_repo=filename,
            repo_id=args.hf_repo,
            repo_type="dataset",
        )
        print(f"Uploaded {filename} to {args.hf_repo}")


if __name__ == "__main__":
    main()
