import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from pathlib import Path
from typing import List
from utils import load_activations, compute_activation_residuals

output_dir = Path(__file__).parent / "output"


def plot_tsne_layers(args, data: dict, layers: List[int], perplexity: int = 30):
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

    output_path = output_dir / f"tsne_{args.file}.png"
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
        args,
        data,
        args.layers,
        perplexity=args.perplexity,
    )


if __name__ == "__main__":
    main()
