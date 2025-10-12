import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
from typing import List, Tuple
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


def plot_tsne_layers(data: dict, layers: List[int], perplexity: int = 30):
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))

    if n_layers == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layers):
        clean_acts, poisoned_acts = compute_activation_residuals(data, layer_idx)
        combined = np.vstack([clean_acts, poisoned_acts])

        tsne = TSNE(n_components=2, perplexity=perplexity)
        embedded = tsne.fit_transform(combined)

        n_clean = len(clean_acts)
        clean_embedded = embedded[:n_clean]
        poisoned_embedded = embedded[n_clean:]

        ax = axes[idx]
        ax.scatter(
            clean_embedded[:, 0],
            clean_embedded[:, 1],
            c='green',
            alpha=0.6,
            label='Clean',
            s=50,
        )
        ax.scatter(
            poisoned_embedded[:, 0],
            poisoned_embedded[:, 1],
            c='red',
            alpha=0.6,
            label='Poisoned',
            s=50,
        )

        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) Layer {layer_idx}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"tsne_layers_{'_'.join(map(str, layers))}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="t-SNE visualization of activations")
    parser.add_argument(
        "--file",
        type=str,
        help="Activations file to load (e.g., activations_100.pt)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 7, 15, 27],
        help="Layer indices to visualize",
    )
    parser.add_argument(
        "--perplexity", type=int, default=30, help="t-SNE perplexity parameter"
    )

    args = parser.parse_args()

    print(f"Loading activations from {args.file}...")
    data = load_activations(args.file)

    print(f"Number of samples: {data['global_metadata']['num_samples']}")
    print(f"Number of layers: {data['global_metadata']['num_layers']}")
    print(f"Model dimension: {data['global_metadata']['d_model']}")
    print(f"Visualizing layers: {args.layers}")

    plot_tsne_layers(
        data,
        args.layers,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
