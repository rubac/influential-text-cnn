"""
Core CNN model architecture for discovering influential text treatments.

Architecture (from the paper, Section 4):
    1. Input: Pre-computed BERT embeddings (N, U, D)
    2. M parallel 1D convolutional layers with different kernel sizes K_l
    3. Sigmoid activation on conv outputs -> phrase filter activations
    4. Max pooling across phrases per filter
    5. Concatenation of pooled activations across conv layers
    6. Fully connected output layer -> sigmoid prediction

Custom loss function (Section 4.3):
    L = BCE + λ_conv_ker * L2(conv_weights)
        + λ_conv_act * max(R)       [activity regularization]
        + λ_out_ker * L1(output_weights)

    where R_{f,g} = max(cor(ã_f, ã_g), 0) for f≠g, 0 for f=g
    and ã_f ∈ R^{N·P} is the vector of filter activations across all phrases
    of all samples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict


class InfluentialTextCNN(nn.Module):
    """
    CNN model for discovering influential text treatments.

    Args:
        embedding_dim: Dimension D of input embeddings.
        num_filters: Number of filters F per convolutional layer.
        kernel_sizes: List of kernel sizes for parallel conv layers.
            M = len(kernel_sizes) parallel layers are created.
        dropout: Dropout rate applied before the output layer.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_filters: int = 8,
        kernel_sizes: Optional[List[int]] = None,
        dropout: float = 0.0,
        task: str = "binary",
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5]
        if task not in ("binary", "continuous"):
            raise ValueError(f"task must be 'binary' or 'continuous', got '{task}'")

        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = sorted(kernel_sizes)
        self.num_conv_layers = len(kernel_sizes)
        self.dropout_rate = dropout
        self.task = task

        # M parallel 1D convolutional layers
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=0,
            )
            for k in self.kernel_sizes
        ])

        # Fully connected output layer
        self._total_filters = num_filters * self.num_conv_layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(self._total_filters, 1)

    def forward(
        self,
        embeddings: torch.Tensor,
        return_activations: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            embeddings: (batch_size, seq_len, embedding_dim) pre-computed
                token embeddings.
            return_activations: If True, return phrase-level activations
                for interpretation and regularization.

        Returns:
            Dict with:
                'logits': (B, 1) raw logits
                'predictions': (B, 1) sigmoid probabilities
                'pooled_activations': (B, total_filters) max-pooled per filter
                'phrase_activations': list of (B, F, P_l) per conv layer
                    [only if return_activations=True]
        """
        # (B, U, D) -> (B, D, U) for Conv1d
        x = embeddings.transpose(1, 2)

        pooled_list = []
        phrase_acts_list = []

        for conv in self.conv_layers:
            # a_{i,f} = σ(W_f · p_i + b) — sigmoid activation per paper
            act = torch.sigmoid(conv(x))  # (B, F, P_l)
            phrase_acts_list.append(act)

            # Max pooling across phrases: a^pooled_{i,f}
            pooled, _ = act.max(dim=2)  # (B, F)
            pooled_list.append(pooled)

        # Concatenate across parallel conv layers: (B, F*M)
        pooled_concat = torch.cat(pooled_list, dim=1)

        # Output layer
        dropped = self.dropout(pooled_concat)
        logits = self.output_layer(dropped)  # (B, 1)

        if self.task == "binary":
            predictions = torch.sigmoid(logits)
        else:
            # Continuous: raw output, no activation
            predictions = logits

        result = {
            'logits': logits,
            'predictions': predictions,
            'pooled_activations': pooled_concat,
        }
        if return_activations:
            result['phrase_activations'] = phrase_acts_list

        return result

    @property
    def total_filters(self) -> int:
        """Total number of filters across all conv layers (F * M)."""
        return self._total_filters

    @property
    def output_weights(self) -> np.ndarray:
        """Output layer weights W^out as numpy array (total_filters,)."""
        return self.output_layer.weight.data.cpu().squeeze(0).numpy()

    def get_conv_weight_info(self) -> List[dict]:
        """Return info about each conv layer's kernel size and weight shape."""
        info = []
        for i, conv in enumerate(self.conv_layers):
            info.append({
                'layer_idx': i,
                'kernel_size': self.kernel_sizes[i],
                'weight_shape': tuple(conv.weight.shape),
                'num_filters': self.num_filters,
            })
        return info


class InfluentialTextLoss(nn.Module):
    """
    Custom loss function combining a data-fidelity term with three
    regularization terms.

    For binary outcomes the data-fidelity term is BCE; for continuous
    outcomes it is MSE.

    Args:
        model: The InfluentialTextCNN model instance.
        lambda_conv_ker: L2 penalty strength on conv layer weights.
        lambda_conv_act: Penalty strength on max pairwise positive correlation
            between filter activations (encourages diverse filters).
        lambda_out_ker: L1 penalty strength on output layer weights.
        pos_weight: Optional positive class weight for BCE (scalar).
            Use for class-imbalanced outcomes.  Ignored when task='continuous'.
        task: 'binary' (BCE loss) or 'continuous' (MSE loss).
    """

    def __init__(
        self,
        model: InfluentialTextCNN,
        lambda_conv_ker: float = 0.001,
        lambda_conv_act: float = 3.0,
        lambda_out_ker: float = 0.0001,
        pos_weight: Optional[float] = None,
        task: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.lambda_conv_ker = lambda_conv_ker
        self.lambda_conv_act = lambda_conv_act
        self.lambda_out_ker = lambda_out_ker
        self.pos_weight = pos_weight
        self.task = task or model.task  # inherit from model if not specified

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        phrase_activations: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full loss.

        Args:
            predictions: (B, 1) sigmoid model outputs.
            targets: (B,) or (B, 1) binary labels.
            phrase_activations: List of (B, F, P_l) per conv layer.
                Required when lambda_conv_act > 0.

        Returns:
            Dict with 'total_loss', 'bce', 'conv_l2', 'act_reg', 'out_l1'.
        """
        device = predictions.device
        targets = targets.float().view(-1, 1)

        # 1) Data-fidelity term
        if self.task == "continuous":
            # Mean squared error for continuous outcomes
            data_loss = F.mse_loss(predictions, targets)
        else:
            # Binary cross-entropy for binary outcomes
            preds = predictions.clamp(1e-7, 1 - 1e-7)
            if self.pos_weight is not None:
                pw = torch.tensor([self.pos_weight], device=device)
                data_loss = F.binary_cross_entropy_with_logits(
                    torch.logit(preds),
                    targets,
                    pos_weight=pw,
                )
            else:
                data_loss = F.binary_cross_entropy(
                    preds.squeeze(1), targets.squeeze(1)
                )

        # 2) L2 on conv weights: Σ (W^conv_{k,d,f})^2
        conv_l2 = torch.tensor(0.0, device=device)
        if self.lambda_conv_ker > 0:
            for conv in self.model.conv_layers:
                conv_l2 = conv_l2 + (conv.weight ** 2).sum()
            conv_l2 = self.lambda_conv_ker * conv_l2

        # 3) Activity regularization: max of non-negative pairwise correlations
        act_reg = torch.tensor(0.0, device=device)
        if self.lambda_conv_act > 0 and phrase_activations is not None:
            for acts in phrase_activations:
                act_reg = act_reg + _activation_correlation_penalty(acts)
            act_reg = self.lambda_conv_act * act_reg

        # 4) L1 on output layer weights: Σ |W^out_f|
        out_l1 = torch.tensor(0.0, device=device)
        if self.lambda_out_ker > 0:
            out_l1 = self.lambda_out_ker * self.model.output_layer.weight.abs().sum()

        total = data_loss + conv_l2 + act_reg + out_l1

        return {
            'total_loss': total,
            'data_loss': data_loss,
            'bce': data_loss,  # backward-compatible alias
            'conv_l2': conv_l2,
            'act_reg': act_reg,
            'out_l1': out_l1,
        }


def _activation_correlation_penalty(activations: torch.Tensor) -> torch.Tensor:
    """
    Compute max non-negative pairwise Pearson correlation between filters.

    Per paper Section 4.3:
        R_{f,g} = max(cor(ã_f, ã_g), 0) for f≠g, 0 for f=g
        penalty = max(R)

    where ã_f ∈ R^{N·P} is the flattened activation vector across all
    phrases of all samples for filter f.

    Args:
        activations: (B, F, P) phrase-level filter activations.

    Returns:
        Scalar: maximum non-negative off-diagonal correlation.
    """
    B, F_dim, P = activations.shape
    if F_dim <= 1:
        return torch.tensor(0.0, device=activations.device)

    # Flatten to (F, B*P)
    flat = activations.permute(1, 0, 2).reshape(F_dim, -1)

    # Center each filter
    centered = flat - flat.mean(dim=1, keepdim=True)
    norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = centered / norms

    # Correlation matrix (F, F)
    corr = torch.mm(normed, normed.t())

    # Zero diagonal, clamp negatives
    mask = 1.0 - torch.eye(F_dim, device=activations.device)
    corr = torch.clamp(corr * mask, min=0.0)

    return corr.max()
