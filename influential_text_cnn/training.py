"""
Training module for the InfluentialTextCNN.

Handles model training with:
- Adam optimizer (Section 4.3)
- Early stopping with patience
- Training/validation split
- Logging of loss components
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
import copy
import logging
import time

from .model import InfluentialTextCNN, InfluentialTextLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Records training metrics per epoch."""
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    bce: List[float] = field(default_factory=list)
    conv_l2: List[float] = field(default_factory=list)
    act_reg: List[float] = field(default_factory=list)
    out_l1: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    training_time_seconds: float = 0.0


class Trainer:
    """
    Trainer for InfluentialTextCNN models.

    Args:
        model: InfluentialTextCNN instance.
        loss_fn: InfluentialTextLoss instance.
        learning_rate: Learning rate for Adam optimizer.
        epochs: Maximum number of training epochs.
        batch_size: Training batch size.
        patience: Early stopping patience (number of epochs without
            improvement before stopping).
        device: Device string.
        verbose: Whether to print training progress.
    """

    def __init__(
        self,
        model: InfluentialTextCNN,
        loss_fn: InfluentialTextLoss,
        learning_rate: float = 0.0001,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.loss_fn.to(self.device)

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_fraction: float = 0.2,
    ) -> TrainingHistory:
        """
        Train the model.

        If validation data is not provided, a random split of val_fraction
        is created from the training data (per paper Section 4.3: "20%
        serving as the validation set").

        Args:
            train_embeddings: (N, U, D) training embeddings.
            train_labels: (N,) binary labels.
            val_embeddings: Optional (N_val, U, D) validation embeddings.
            val_labels: Optional (N_val,) validation labels.
            val_fraction: Fraction of training data to use as validation
                if val_embeddings is not provided.

        Returns:
            TrainingHistory with per-epoch metrics.
        """
        # Create validation split if needed
        if val_embeddings is None:
            N = len(train_embeddings)
            perm = np.random.permutation(N)
            n_val = int(N * val_fraction)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]

            val_embeddings = train_embeddings[val_idx]
            val_labels = train_labels[val_idx]
            train_embeddings = train_embeddings[train_idx]
            train_labels = train_labels[train_idx]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(train_embeddings, dtype=torch.float32),
            torch.tensor(train_labels, dtype=torch.float32),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        val_dataset = TensorDataset(
            torch.tensor(val_embeddings, dtype=torch.float32),
            torch.tensor(val_labels, dtype=torch.float32),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        # Training loop
        history = TrainingHistory()
        best_model_state = None
        patience_counter = 0
        start_time = time.time()

        for epoch in range(self.epochs):
            # --- Train ---
            self.model.train()
            epoch_losses = []
            epoch_bce = []
            epoch_conv_l2 = []
            epoch_act_reg = []
            epoch_out_l1 = []
            epoch_correct = 0
            epoch_total = 0

            for emb_batch, label_batch in train_loader:
                emb_batch = emb_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                optimizer.zero_grad()

                # Forward with activations for regularization
                output = self.model(emb_batch, return_activations=True)

                loss_dict = self.loss_fn(
                    output['predictions'],
                    label_batch,
                    phrase_activations=output['phrase_activations'],
                )

                loss_dict['total_loss'].backward()
                optimizer.step()

                batch_size = emb_batch.size(0)
                epoch_losses.append(loss_dict['total_loss'].item() * batch_size)
                epoch_bce.append(loss_dict['bce'].item() * batch_size)
                epoch_conv_l2.append(loss_dict['conv_l2'].item() * batch_size)
                epoch_act_reg.append(loss_dict['act_reg'].item() * batch_size)
                epoch_out_l1.append(loss_dict['out_l1'].item() * batch_size)

                preds_binary = (output['predictions'].squeeze() > 0.5).float()
                epoch_correct += (preds_binary == label_batch).sum().item()
                epoch_total += batch_size

            n_train = epoch_total
            history.train_loss.append(sum(epoch_losses) / n_train)
            history.bce.append(sum(epoch_bce) / n_train)
            history.conv_l2.append(sum(epoch_conv_l2) / n_train)
            history.act_reg.append(sum(epoch_act_reg) / n_train)
            history.out_l1.append(sum(epoch_out_l1) / n_train)

            if self.model.task == "binary":
                history.train_acc.append(epoch_correct / n_train)
            else:
                # For continuous outcomes, track negative MSE as "accuracy"
                # (higher is better, for consistent early-stopping logic)
                history.train_acc.append(-sum(epoch_bce) / n_train)

            # --- Validate ---
            self.model.eval()
            val_losses = []
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for emb_batch, label_batch in val_loader:
                    emb_batch = emb_batch.to(self.device)
                    label_batch = label_batch.to(self.device)

                    output = self.model(emb_batch, return_activations=True)
                    loss_dict = self.loss_fn(
                        output['predictions'],
                        label_batch,
                        phrase_activations=output['phrase_activations'],
                    )

                    batch_size = emb_batch.size(0)
                    val_losses.append(loss_dict['total_loss'].item() * batch_size)

                    if self.model.task == "binary":
                        preds_binary = (output['predictions'].squeeze() > 0.5).float()
                        val_correct += (preds_binary == label_batch).sum().item()
                    else:
                        # Accumulate MSE contribution for continuous
                        mse_contrib = F.mse_loss(
                            output['predictions'].squeeze(), label_batch,
                            reduction='sum'
                        ).item()
                        val_correct += mse_contrib  # reused as sum-of-squared-error
                    val_total += batch_size

            val_loss = sum(val_losses) / val_total
            if self.model.task == "binary":
                val_acc = val_correct / val_total
            else:
                val_acc = -(val_correct / val_total)  # negative MSE
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)

            # --- Early stopping ---
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                if self.model.task == "binary":
                    logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {history.train_loss[-1]:.4f} "
                        f"(BCE: {history.bce[-1]:.4f}) | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Train Acc: {history.train_acc[-1]:.3f} | "
                        f"Val Acc: {val_acc:.3f}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch:3d} | "
                        f"Train Loss: {history.train_loss[-1]:.4f} "
                        f"(MSE: {history.bce[-1]:.4f}) | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Train MSE: {-history.train_acc[-1]:.4f} | "
                        f"Val MSE: {-val_acc:.4f}"
                    )

            if patience_counter >= self.patience:
                if self.verbose:
                    logger.info(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch: {history.best_epoch}"
                    )
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        history.training_time_seconds = time.time() - start_time
        if self.verbose:
            logger.info(
                f"Training complete in {history.training_time_seconds:.1f}s. "
                f"Best val loss: {history.best_val_loss:.4f} at epoch "
                f"{history.best_epoch}"
            )

        return history

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        For binary tasks: returns 'accuracy', 'f1', 'precision', 'recall'.
        For continuous tasks: returns 'mse', 'rmse', 'r2', 'mae'.
        """
        self.model.eval()
        dataset = TensorDataset(
            torch.tensor(embeddings, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for emb_batch, label_batch in loader:
                emb_batch = emb_batch.to(self.device)
                output = self.model(emb_batch)
                all_preds.append(output['predictions'].squeeze().cpu().numpy())
                all_labels.append(label_batch.numpy())

        preds = np.concatenate(all_preds).ravel()
        labels = np.concatenate(all_labels).ravel()

        if self.model.task == "continuous":
            mse = float(np.mean((preds - labels) ** 2))
            ss_res = np.sum((labels - preds) ** 2)
            ss_tot = np.sum((labels - labels.mean()) ** 2)
            r2 = 1 - ss_res / max(ss_tot, 1e-8)
            return {
                'mse': mse,
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2),
                'mae': float(np.mean(np.abs(preds - labels))),
            }
        else:
            preds_binary = (preds > 0.5).astype(float)
            acc = np.mean(preds_binary == labels)
            tp = np.sum((preds_binary == 1) & (labels == 1))
            fp = np.sum((preds_binary == 1) & (labels == 0))
            fn = np.sum((preds_binary == 0) & (labels == 1))

            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)

            return {
                'accuracy': float(acc),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }
