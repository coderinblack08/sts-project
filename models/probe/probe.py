import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from jaxtyping import Float
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import load_activations, compute_activation_residuals

output_dir = Path(__file__).parent / "output"


def prepare_data(
    data: dict, layer_idx: int
) -> Tuple[Float[np.ndarray, "n_samples d_model"], Float[np.ndarray, "n_samples"]]:
    clean_residuals, poisoned_residuals = compute_activation_residuals(data, layer_idx)
    
    X = np.vstack([clean_residuals, poisoned_residuals])
    y = np.hstack([np.zeros(len(clean_residuals)), np.ones(len(poisoned_residuals))])
    
    return X, y


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
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


def train_and_evaluate_probe(
    train_data: dict, test_data: dict, layer_idx: int
) -> Dict[str, float]:
    X_train, y_train = prepare_data(train_data, layer_idx)
    X_test, y_test = prepare_data(test_data, layer_idx)
    
    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)
    
    y_pred = probe.predict(X_test)
    y_proba = probe.predict_proba(X_test)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train logistic regression probes for clean vs poisoned classification"
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
    
    results = {}
    for layer_idx in args.layers:
        print(f"Training probe for layer {layer_idx}...")
        metrics = train_and_evaluate_probe(train_data, test_data, layer_idx)
        results[layer_idx] = metrics
        
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
    
    best_layer = max(results.keys(), key=lambda k: results[k]['auroc'])
    print(f"Best performing layer: {best_layer} (AUROC: {results[best_layer]['auroc']:.4f})")


if __name__ == "__main__":
    main()

