import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from jaxtyping import Float
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils import load_activations, compute_activation_residuals

output_dir = Path(__file__).parent / "output"


def prepare_data(
    data: dict, layer_idx: int
) -> Tuple[Float[np.ndarray, "n_samples d_model"], Float[np.ndarray, "n_samples"]]:
    clean_residuals, poisoned_residuals = compute_activation_residuals(data, layer_idx)

    X = np.vstack([clean_residuals, poisoned_residuals])
    y = np.hstack([np.zeros(len(clean_residuals)), np.ones(len(poisoned_residuals))])

    return X, y


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "auroc": roc_auc_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
    }

    return metrics


def train_mass_mean_probe(
    X_train: Float[np.ndarray, "n_samples d_model"],
    y_train: Float[np.ndarray, "n_samples"],
) -> Tuple[Float[np.ndarray, "d_model"], Float[np.ndarray, "d_model d_model"]]:
    mu_pos = X_train[y_train == 1].mean(axis=0)
    mu_neg = X_train[y_train == 0].mean(axis=0)
    theta_mm = mu_pos - mu_neg

    X_pos_centered = X_train[y_train == 1] - mu_pos
    X_neg_centered = X_train[y_train == 0] - mu_neg
    X_centered = np.vstack([X_neg_centered, X_pos_centered])

    Sigma = np.cov(X_centered.T)

    d = X_train.shape[1]
    epsilon = 1e-6
    Sigma_inv = np.linalg.inv(Sigma + epsilon * np.eye(d))

    return theta_mm, Sigma_inv


def predict_mass_mean(
    X: Float[np.ndarray, "n_samples d_model"],
    theta_mm: Float[np.ndarray, "d_model"],
    Sigma_inv: Float[np.ndarray, "d_model d_model"],
) -> Float[np.ndarray, "n_samples"]:
    logits = X @ Sigma_inv @ theta_mm
    proba = expit(logits)
    return proba


def train_and_evaluate_mass_mean(
    train_data: dict, test_data: dict, layer_idx: int
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:
    X_train, y_train = prepare_data(train_data, layer_idx)
    X_test, y_test = prepare_data(test_data, layer_idx)

    theta_mm, Sigma_inv = train_mass_mean_probe(X_train, y_train)

    y_proba = predict_mass_mean(X_test, theta_mm, Sigma_inv)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_metrics(y_test, y_pred, y_proba)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return metrics, (fpr, tpr)


def plot_all_roc_curves(
    layer_data: Dict[int, Tuple[np.ndarray, np.ndarray, float]],
    output_filename: str,
    format: str,
):
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (layer_idx, (fpr, tpr, auroc)) in enumerate(sorted(layer_data.items())):
        color = colors[idx % len(colors)]
        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            color=color,
            label=f'Layer {layer_idx} (area = {auroc:.4f})',
        )

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, bbox_inches='tight', format=format)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train mass-mean probes for clean vs poisoned classification"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="Training activations file (e.g., activations_1000.pt)",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test activations file (e.g., activations_test_100.pt)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[0, 7, 15, 27],
        help="Layer indices to train probes on",
    )

    args = parser.parse_args()

    train_data = load_activations(args.train_file)
    test_data = load_activations(args.test_file)

    print(f"Training set: {train_data['global_metadata']['num_samples']} samples")
    print(f"Test set: {test_data['global_metadata']['num_samples']} samples")
    print()

    results = {}
    roc_data = {}

    for layer_idx in args.layers:
        print(f"Training mass-mean probe for layer {layer_idx}...")
        metrics, (fpr, tpr) = train_and_evaluate_mass_mean(
            train_data, test_data, layer_idx
        )
        results[layer_idx] = metrics
        roc_data[layer_idx] = (fpr, tpr, metrics['auroc'])

        print(f"Layer {layer_idx} results:")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  TPR:       {metrics['tpr']:.4f}")
        print(f"  TNR:       {metrics['tnr']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f}")
        print(f"  FNR:       {metrics['fnr']:.4f}")
        print()

    output_filename = output_dir / "roc_curves_mass_mean.png"
    plot_all_roc_curves(roc_data, str(output_filename), format="png")
    eps_filename = output_dir / "roc_curves_mass_mean.eps"
    plot_all_roc_curves(roc_data, str(eps_filename), format="eps")

    best_layer = max(results.keys(), key=lambda k: results[k]['auroc'])
    print(
        f"Best performing layer: {best_layer} (AUROC: {results[best_layer]['auroc']:.4f})"
    )


if __name__ == "__main__":
    main()
