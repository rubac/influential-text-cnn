"""
Hyperparameter tuning module.

Implements k-fold cross-validation grid search over model hyperparameters,
following Section 4.3 of the paper:
    - 5-fold CV on the training set
    - Model selection based on accuracy, filter diversity, and useful filters
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from itertools import product
import logging
import copy

from .model import InfluentialTextCNN, InfluentialTextLoss
from .training import Trainer, TrainingHistory

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result from one hyperparameter configuration."""
    params: Dict[str, Any]
    val_accuracy_mean: float
    val_accuracy_std: float
    val_loss_mean: float
    val_loss_std: float
    num_useful_filters_mean: float
    max_correlation_mean: float
    combined_score: float = 0.0


@dataclass
class TuningReport:
    """Full tuning report."""
    best_params: Dict[str, Any]
    best_result: TuningResult
    all_results: List[TuningResult]
    n_configs_tried: int = 0


# Default parameter grid matching the paper's search space
DEFAULT_PARAM_GRID = {
    'num_filters': [4, 8, 16],
    'kernel_sizes': [[5], [7], [5, 7]],
    'lambda_conv_ker': [0.0, 0.0001, 0.001],
    'lambda_conv_act': [0.0, 1.0, 3.0],
    'lambda_out_ker': [0.0001, 0.001, 0.01],
    'learning_rate': [0.00001, 0.0001, 0.001],
}


def _count_useful_filters(model, embeddings, device, threshold=0.05):
    """Count filters with activation range >= threshold."""
    model.eval()
    model.to(device)
    all_pooled = []
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            end = min(start + batch_size, len(embeddings))
            emb = torch.tensor(
                embeddings[start:end], dtype=torch.float32
            ).to(device)
            out = model(emb)
            all_pooled.append(out['pooled_activations'].cpu().numpy())

    pooled = np.concatenate(all_pooled, axis=0)
    ranges = pooled.max(axis=0) - pooled.min(axis=0)
    return int((ranges >= threshold).sum())


def _max_filter_correlation(model, embeddings, device):
    """Compute max non-negative pairwise filter correlation."""
    model.eval()
    model.to(device)
    all_acts = [[] for _ in model.conv_layers]

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(embeddings), batch_size):
            end = min(start + batch_size, len(embeddings))
            emb = torch.tensor(
                embeddings[start:end], dtype=torch.float32
            ).to(device)
            out = model(emb, return_activations=True)
            for l, acts in enumerate(out['phrase_activations']):
                all_acts[l].append(acts.cpu().numpy())

    max_corr = 0.0
    for l in range(len(model.conv_layers)):
        acts = np.concatenate(all_acts[l], axis=0)  # (N, F, P_l)
        F = acts.shape[1]
        flat = acts.transpose(1, 0, 2).reshape(F, -1)  # (F, N*P_l)
        if flat.shape[1] < 2:
            continue
        corr = np.corrcoef(flat)
        if np.any(np.isnan(corr)):
            continue
        np.fill_diagonal(corr, 0.0)
        corr_nn = np.clip(corr, 0, None)
        max_corr = max(max_corr, float(corr_nn.max()))

    return max_corr


def tune_hyperparameters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    param_grid: Optional[Dict[str, List]] = None,
    n_folds: int = 5,
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 15,
    device: str = "cpu",
    selection_method: str = "combined",
    task: str = "binary",
) -> TuningReport:
    """
    Perform hyperparameter tuning via k-fold cross-validation.

    Args:
        embeddings: (N, U, D) training set embeddings.
        labels: (N,) outcome labels.
        param_grid: Dict mapping parameter names to lists of values.
            If None, uses DEFAULT_PARAM_GRID.
        n_folds: Number of CV folds (paper uses 5).
        epochs: Max epochs per training run.
        batch_size: Batch size.
        patience: Early stopping patience.
        device: Device string.
        selection_method: 'combined' (acc + diversity), 'accuracy', or 'loss'.
        task: 'binary' or 'continuous'.

    Returns:
        TuningReport with best parameters and all results.
    """
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID

    embedding_dim = embeddings.shape[2]

    # Generate all parameter combinations
    param_names = sorted(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]
    all_combos = list(product(*param_values))

    # Filter out invalid combinations (e.g., too many total filters)
    valid_combos = []
    for combo in all_combos:
        params = dict(zip(param_names, combo))
        nf = params.get('num_filters', 8)
        ks = params.get('kernel_sizes', [5])
        total = nf * len(ks)
        if total <= 32:
            valid_combos.append(params)

    logger.info(
        f"Tuning: {len(valid_combos)} configurations, "
        f"{n_folds}-fold CV each"
    )

    # Create CV folds
    if task == "binary":
        from sklearn.model_selection import StratifiedKFold
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices = list(kfold.split(embeddings, labels))
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_indices = list(kfold.split(embeddings))

    all_results = []

    for config_idx, params in enumerate(valid_combos):
        if (config_idx + 1) % 20 == 0 or config_idx == 0:
            logger.info(f"  Config {config_idx + 1}/{len(valid_combos)}...")

        nf = params.get('num_filters', 8)
        ks = params.get('kernel_sizes', [5])
        lck = params.get('lambda_conv_ker', 0.001)
        lca = params.get('lambda_conv_act', 1.0)
        lok = params.get('lambda_out_ker', 0.0001)
        lr = params.get('learning_rate', 0.0001)

        fold_accs = []
        fold_losses = []
        fold_useful = []
        fold_corrs = []

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                model = InfluentialTextCNN(
                    embedding_dim=embedding_dim,
                    num_filters=nf,
                    kernel_sizes=ks,
                    task=task,
                )
                loss_fn = InfluentialTextLoss(
                    model=model,
                    lambda_conv_ker=lck,
                    lambda_conv_act=lca,
                    lambda_out_ker=lok,
                )
                trainer = Trainer(
                    model=model,
                    loss_fn=loss_fn,
                    learning_rate=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    patience=patience,
                    device=device,
                    verbose=False,
                )

                history = trainer.fit(
                    train_embeddings=embeddings[train_idx],
                    train_labels=labels[train_idx],
                    val_embeddings=embeddings[val_idx],
                    val_labels=labels[val_idx],
                )

                # Evaluate
                metrics = trainer.evaluate(embeddings[val_idx], labels[val_idx])
                if task == "binary":
                    fold_accs.append(metrics['accuracy'])
                else:
                    # For continuous: use negative MSE (higher is better)
                    fold_accs.append(-metrics['mse'])
                fold_losses.append(history.best_val_loss)

                # Count useful filters and correlation
                n_useful = _count_useful_filters(model, embeddings[val_idx], device)
                fold_useful.append(n_useful)

                max_corr = _max_filter_correlation(model, embeddings[val_idx], device)
                fold_corrs.append(max_corr)

        except Exception as e:
            logger.debug(f"  Config {config_idx + 1} failed: {e}")
            continue

        result = TuningResult(
            params=params,
            val_accuracy_mean=float(np.mean(fold_accs)),
            val_accuracy_std=float(np.std(fold_accs)),
            val_loss_mean=float(np.mean(fold_losses)),
            val_loss_std=float(np.std(fold_losses)),
            num_useful_filters_mean=float(np.mean(fold_useful)),
            max_correlation_mean=float(np.mean(fold_corrs)),
        )

        # Combined score (paper: "more subjective" — balances accuracy,
        # filter diversity, and number of useful filters)
        total_possible = nf * len(ks)
        useful_frac = result.num_useful_filters_mean / max(total_possible, 1)
        result.combined_score = (
            result.val_accuracy_mean
            + 0.3 * useful_frac
            - 0.3 * result.max_correlation_mean
        )

        all_results.append(result)

    if not all_results:
        raise RuntimeError("All configurations failed during tuning.")

    # Select best
    if selection_method == "combined":
        best_result = max(all_results, key=lambda r: r.combined_score)
    elif selection_method == "accuracy":
        best_result = max(all_results, key=lambda r: r.val_accuracy_mean)
    elif selection_method == "loss":
        best_result = min(all_results, key=lambda r: r.val_loss_mean)
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")

    logger.info(
        f"\n  Best config: {best_result.params}\n"
        f"  Val accuracy: {best_result.val_accuracy_mean:.4f} "
        f"± {best_result.val_accuracy_std:.4f}\n"
        f"  Useful filters: {best_result.num_useful_filters_mean:.1f}\n"
        f"  Max correlation: {best_result.max_correlation_mean:.4f}"
    )

    return TuningReport(
        best_params=best_result.params,
        best_result=best_result,
        all_results=all_results,
        n_configs_tried=len(valid_combos),
    )
