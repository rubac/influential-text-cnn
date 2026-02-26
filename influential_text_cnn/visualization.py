"""
Visualization utilities for interpreting and presenting results.

Includes:
- Filter activation correlation grids (Figure 4 in the paper)
- Model fit comparison bar charts (Figures 2, 3)
- Conditional density plots for filter activations (Table 10)
- Cross-method correlation plots (Figures 5-8)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from .interpretation import InterpretationResult
from .benchmarks import BenchmarkResult


def plot_filter_correlation_grid(
    result: InterpretationResult,
    only_active: bool = False,
    title: str = "Filter Activation Correlations",
    figsize: Tuple[int, int] = (8, 7),
    save_path: Optional[str] = None,
):
    """
    Plot pairwise correlation grid between filter pooled activations.

    Reproduces Figure 4 from the paper: dark red = correlation near 1,
    dark blue = near -1, white = near 0.

    Args:
        result: InterpretationResult from interpretation.
        only_active: Only show active filters.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    pooled = result.pooled_activations
    n_filters = pooled.shape[1]

    if only_active:
        active_mask = np.array([f.is_active for f in result.filters])
        # Sort filters by original index
        active_indices = sorted(
            [f.filter_idx for f in result.filters if f.is_active]
        )
        pooled = pooled[:, active_indices]
        labels = [f"F{i}" for i in active_indices]
    else:
        labels = [f"F{i}" for i in range(n_filters)]

    corr = np.corrcoef(pooled.T)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    im = ax.imshow(corr, cmap=cmap, norm=norm, aspect='equal')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=12)

    plt.colorbar(im, ax=ax, label="Correlation", shrink=0.8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_model_comparison(
    cnn_r2: float,
    cnn_mse: float,
    benchmarks: Dict[str, BenchmarkResult],
    cnn_r2_ci: Optional[Tuple[float, float]] = None,
    cnn_mse_ci: Optional[Tuple[float, float]] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None,
):
    """
    Bar chart comparing R²_adj and MSE across methods.

    Reproduces Figures 2 and 3 from the paper.

    Args:
        cnn_r2, cnn_mse: CNN model metrics.
        benchmarks: Dict of benchmark name -> BenchmarkResult.
        cnn_r2_ci, cnn_mse_ci: Optional confidence intervals for CNN.
        title: Plot title.
        figsize: Figure size.
        save_path: Save path.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    methods = ['CNN'] + list(benchmarks.keys())
    r2_values = [cnn_r2] + [b.r_squared_adj for b in benchmarks.values()]
    mse_values = [cnn_mse] + [b.mse for b in benchmarks.values()]

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'][:len(methods)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # R²_adj plot
    bars1 = ax1.bar(methods, r2_values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars1, r2_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax1.set_ylabel('R²_adj', fontsize=11)
    ax1.set_title('Adjusted R-squared', fontsize=12)
    ax1.set_ylim(0, max(r2_values) * 1.3 + 0.05)

    # MSE plot
    bars2 = ax2.bar(methods, mse_values, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars2, mse_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('Mean Squared Error', fontsize=12)
    ax2.set_ylim(0, max(mse_values) * 1.3 + 0.05)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_filter_summary(
    result: InterpretationResult,
    max_filters: int = 16,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
):
    """
    Horizontal bar chart showing output weights and effect estimates
    for each active filter, with top phrase annotations.

    Args:
        result: InterpretationResult.
        max_filters: Max filters to show.
        figsize: Figure size.
        save_path: Save path.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    active = [f for f in result.filters if f.is_active][:max_filters]
    if not active:
        print("No active filters to plot.")
        return

    # Sort by output weight
    active.sort(key=lambda f: f.output_weight, reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    names = [f"F{f.filter_idx}" for f in active]
    w_out = [f.output_weight for f in active]
    betas = [f.effect_estimate if f.effect_estimate is not None else 0 for f in active]
    cis = [f.effect_ci for f in active]

    y_pos = range(len(active))

    # Output weights
    colors_w = ['#e74c3c' if w > 0 else '#3498db' for w in w_out]
    ax1.barh(y_pos, w_out, color=colors_w, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('W^out', fontsize=11)
    ax1.set_title('Output Layer Weights', fontsize=12)
    ax1.axvline(x=0, color='black', linewidth=0.5)

    # Effect estimates with CIs
    colors_b = ['#e74c3c' if b > 0 else '#3498db' for b in betas]
    ax2.barh(y_pos, betas, color=colors_b, alpha=0.8, edgecolor='black')
    for i, (ci, b) in enumerate(zip(cis, betas)):
        if ci is not None:
            ax2.plot([ci[0], ci[1]], [i, i], 'k-', linewidth=1.5)
    ax2.set_xlabel('β (OLS estimate)', fontsize=11)
    ax2.set_title('Treatment Effect Estimates', fontsize=12)
    ax2.axvline(x=0, color='black', linewidth=0.5)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=9)
    ax1.invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    """
    Plot training curves: loss, accuracy, and loss components.

    Args:
        history: TrainingHistory object.
        figsize: Figure size.
        save_path: Save path.
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(len(history.train_loss))

    # Total loss
    axes[0].plot(epochs, history.train_loss, label='Train', color='#e74c3c')
    axes[0].plot(epochs, history.val_loss, label='Val', color='#3498db')
    axes[0].axvline(x=history.best_epoch, color='green', linestyle='--',
                    alpha=0.7, label=f'Best (epoch {history.best_epoch})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend(fontsize=8)

    # Accuracy
    axes[1].plot(epochs, history.train_acc, label='Train', color='#e74c3c')
    axes[1].plot(epochs, history.val_acc, label='Val', color='#3498db')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend(fontsize=8)

    # Loss components
    axes[2].plot(epochs, history.bce, label='BCE', color='#e74c3c')
    axes[2].plot(epochs, history.conv_l2, label='Conv L2', color='#3498db')
    axes[2].plot(epochs, history.act_reg, label='Act Reg', color='#2ecc71')
    axes[2].plot(epochs, history.out_l1, label='Out L1', color='#9b59b6')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss Component')
    axes[2].set_title('Loss Components')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
