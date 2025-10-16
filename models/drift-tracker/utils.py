import json
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from jaxtyping import Float
from huggingface_hub import hf_hub_download

dataset_dir = Path(__file__).parent / "dataset"
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)


def load_samples(n: Optional[int], split: str):
    with open(dataset_dir / f"injection_{split}_min.json", "r") as f:
        data = json.load(f)
    data = [(i, sample) for i, sample in enumerate(data)]
    return data if n is None else data[:n]


def load_activations(
    filename: str, hf: bool = False, hf_repo: Optional[str] = None
) -> dict:
    if hf:
        if hf_repo is None:
            raise ValueError(f"Repo {hf_repo} no found.")
        filepath = hf_hub_download(
            repo_id=hf_repo, filename=filename, local_files_only=True
        )
        return torch.load(filepath)
    else:
        filepath = output_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found.")
        return torch.load(filepath)


def extract_layer_activations(
    data: dict, layer_idx: int
) -> Tuple[
    Float[np.ndarray, "n_samples d_model"], Float[np.ndarray, "n_samples d_model"]
]:
    clean_acts = []
    poisoned_acts = []

    for sample in data["samples"]:
        clean_positions = sample["activations"]["clean"]
        poisoned_positions = sample["activations"]["poisoned"]
        positions = sample["positions"]

        primary_key = f"pos_{positions['end_of_primary']}"
        clean_key = f"pos_{positions['end_of_clean']}"
        poisoned_key = f"pos_{positions['end_of_poisoned']}"

        clean_acts.append(
            [
                clean_positions[primary_key][layer_idx].numpy(),
                clean_positions[clean_key][layer_idx].numpy(),
            ]
        )
        poisoned_acts.append(
            [
                clean_positions[primary_key][layer_idx].numpy(),
                poisoned_positions[poisoned_key][layer_idx].numpy(),
            ]
        )

    return np.array(clean_acts), np.array(poisoned_acts)


def compute_activation_residuals(
    data: dict, layer_idx: int
) -> Tuple[
    Float[np.ndarray, "n_samples d_model"], Float[np.ndarray, "n_samples d_model"]
]:
    clean_residuals = []
    poisoned_residuals = []

    activations = extract_layer_activations(data, layer_idx)

    for clean_act, poisoned_act in zip(activations[0], activations[1]):
        clean_residuals.append(clean_act[1] - clean_act[0])
        poisoned_residuals.append(poisoned_act[1] - poisoned_act[0])

    return np.array(clean_residuals), np.array(poisoned_residuals)
