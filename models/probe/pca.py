import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from utils import (
    compute_activation_residuals,
    extract_layer_activations,
    load_activations,
)

output_dir = Path(__file__).parent / "output"


def plot_pca_layers(args, data: dict, layers: List[int], use_residuals: bool = False):
    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5))

    if n_layers == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layers):
        if use_residuals:
            clean_data, poisoned_data = compute_activation_residuals(data, layer_idx)
        else:
            activations = extract_layer_activations(data, layer_idx)
            clean_data = activations[0][:, 1]
            poisoned_data = activations[1][:, 1]

        combined = np.vstack([clean_data, poisoned_data])

        pca = PCA(n_components=2)
        embedded = pca.fit_transform(combined)

        n_clean = len(clean_data)
        clean_embedded = embedded[:n_clean]
        poisoned_embedded = embedded[n_clean:]

        ax = axes[idx]
        ax.scatter(
            poisoned_embedded[:, 0],
            poisoned_embedded[:, 1],
            facecolors='none',
            edgecolors='red',
            linewidths=1.5,
            label='Poisoned',
            s=50,
            marker='^',
        )
        ax.scatter(
            clean_embedded[:, 0],
            clean_embedded[:, 1],
            facecolors='none',
            edgecolors='green',
            linewidths=1.5,
            label='Clean',
            s=50,
            marker='^',
        )

        variance = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({variance[0]:.1%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({variance[1]:.1%})', fontsize=12)
        ax.set_title(f'({chr(97+idx)}) Layer {layer_idx}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    suffix = "residuals" if use_residuals else "activations"
    output_path_png = output_dir / f"pca_{suffix}_{args.file}.png"
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', format='png')
    print(f"Saved plot to {output_path_png}")

    output_path_eps = output_dir / f"pca_{suffix}_{args.file}.eps"
    plt.savefig(output_path_eps, dpi=300, bbox_inches='tight', format='eps')
    print(f"Saved plot to {output_path_eps}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PCA visualization of activations")
    parser.add_argument("--hf", action="store_true")
    parser.add_argument("--hf-repo", type=str)
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
        "--residuals",
        action="store_true",
        help="Use activation residuals instead of raw activations",
    )
    args = parser.parse_args()

    print(f"Loading activations from {args.file}...")
    data = load_activations(args.file, args.hf, args.hf_repo)

    if args.residuals:
        print("Computing PCA on activation residuals...")
        plot_pca_layers(args, data, args.layers, use_residuals=True)
    else:
        print("Computing PCA on raw activations...")
        plot_pca_layers(args, data, args.layers, use_residuals=False)


if __name__ == "__main__":
    main()
