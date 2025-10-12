import torch
import numpy as np
from pathlib import Path
from typing import Tuple
from jaxtyping import Float

output_dir = Path(__file__).parent / "output"


def load_activations(filename: str) -> dict:
    filepath = output_dir / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} not found. Run activations.py first.")
    return torch.load(filepath)


def extract_layer_activations(
    data: dict, layer_idx: int
) -> Tuple[
    Float[np.ndarray, "n_samples d_model"], Float[np.ndarray, "n_samples d_model"]
]:
    clean_acts = []
    poisoned_acts = []

    for sample in data["samples"]:
        clean_positions = sample["clean"]
        poisoned_positions = sample["poisoned"]
        metadata = sample["metadata"]

        primary_key = f"pos_{metadata['end_of_primary_token']}"
        injection_key = f"pos_{metadata['end_of_injection_token']}"
        clean_key = f"pos_{metadata['end_of_clean_token']}"
        poisoned_key = f"pos_{metadata['end_of_poisoned_token']}"

        clean_acts.append(
            [
                clean_positions[primary_key][layer_idx].numpy(),
                clean_positions[clean_key][layer_idx].numpy(),
            ]
        )
        poisoned_acts.append(
            [
                poisoned_positions[primary_key][layer_idx].numpy(),
                poisoned_positions[injection_key][layer_idx].numpy(),
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
        poisoned_residuals.append(poisoned_act[2] - poisoned_act[0])

    return np.array(clean_residuals), np.array(poisoned_residuals)

