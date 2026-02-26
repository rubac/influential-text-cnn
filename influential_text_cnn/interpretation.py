"""
Filter interpretation module.

Per Section 4.4, interpretation uses three components:
1. Filter activations of each phrase for each filter (a_{i,f})
2. Output layer weights (W^out)
3. Original input text samples (T_i)

The process identifies the most highly-activating phrases per filter
to understand what textual patterns each filter has learned to detect.

Also includes treatment effect estimation via OLS regression on the
test set (Section 3), and utility functions for producing the summary
tables shown in the paper (Tables 1, 2).
"""

import numpy as np
import torch
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterInfo:
    """Information about a single learned filter."""
    filter_idx: int                    # global filter index
    conv_layer_idx: int                # which conv layer
    local_filter_idx: int              # index within the conv layer
    kernel_size: int                   # kernel size of the conv layer
    output_weight: float               # W^out_f
    activation_range: float            # max - min of pooled activations
    is_active: bool                    # whether range > threshold
    top_phrases: List[dict] = field(default_factory=list)
    # Each phrase dict: {
    #     'text': str,              # reconstructed phrase text
    #     'tokens': List[str],      # raw tokens
    #     'activation': float,      # filter activation value
    #     'sample_idx': int,        # which text sample
    #     'phrase_position': int,   # position within the sample
    #     'sample_text': str,       # full text of the sample for context
    # }
    manual_label: Optional[str] = None
    effect_estimate: Optional[float] = None
    effect_ci: Optional[Tuple[float, float]] = None


@dataclass
class InterpretationResult:
    """Full interpretation results for a trained model."""
    filters: List[FilterInfo]
    output_weights: np.ndarray
    pooled_activations: np.ndarray   # (N_test, total_filters)
    effect_estimates: Optional[np.ndarray] = None
    effect_cis: Optional[np.ndarray] = None
    r_squared_adj: Optional[float] = None


class FilterInterpreter:
    """
    Interprets a trained InfluentialTextCNN model by extracting the top
    activating phrases per filter and estimating treatment effects.

    Args:
        model: Trained InfluentialTextCNN model.
        activation_threshold: Filters with pooled activation range below
            this threshold are considered "inactive" (default 0.05 per paper).
        top_k_phrases: Number of top phrases to extract per filter.
        deduplicate_phrases: Whether to skip duplicate phrase texts.
    """

    def __init__(
        self,
        model,
        activation_threshold: float = 0.05,
        top_k_phrases: int = 5,
        deduplicate_phrases: bool = True,
    ):
        self.model = model
        self.activation_threshold = activation_threshold
        self.top_k_phrases = top_k_phrases
        self.deduplicate_phrases = deduplicate_phrases

    def interpret(
        self,
        embeddings: np.ndarray,
        tokens: List[List[str]],
        labels: np.ndarray,
        texts: Optional[List[str]] = None,
        n_bootstrap: int = 1000,
        estimate_effects: bool = True,
        device: str = "cpu",
    ) -> InterpretationResult:
        """
        Run full interpretation on a dataset (typically the test set).

        Args:
            embeddings: (N, U, D) pre-computed BERT embeddings.
            tokens: List of N lists of token strings.
            labels: (N,) binary outcome labels.
            texts: Optional list of N original text strings for context.
            n_bootstrap: Number of bootstrap samples for effect CIs.
            estimate_effects: Whether to run OLS effect estimation.
            device: Device for model inference.

        Returns:
            InterpretationResult with filter info and effect estimates.
        """
        self.model.eval()
        self.model.to(device)

        N = len(embeddings)
        output_weights = self.model.output_weights  # (total_filters,)

        # --- Compute activations ---
        all_pooled = []
        all_phrase_acts = []  # list of lists (per conv layer)

        batch_size = 128
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                emb_batch = torch.tensor(
                    embeddings[start:end], dtype=torch.float32
                ).to(device)

                result = self.model(emb_batch, return_activations=True)
                all_pooled.append(
                    result['pooled_activations'].cpu().numpy()
                )
                # phrase_activations: list of (B, F, P_l) per conv layer
                batch_phrase_acts = [
                    a.cpu().numpy() for a in result['phrase_activations']
                ]
                all_phrase_acts.append(batch_phrase_acts)

        # Concatenate across batches
        pooled = np.concatenate(all_pooled, axis=0)  # (N, total_filters)

        # Merge phrase activations across batches
        n_conv = len(all_phrase_acts[0])
        phrase_acts_per_layer = []
        for layer_idx in range(n_conv):
            layer_acts = np.concatenate(
                [batch[layer_idx] for batch in all_phrase_acts],
                axis=0,
            )  # (N, F, P_l)
            phrase_acts_per_layer.append(layer_acts)

        # --- Build FilterInfo for each filter ---
        filters = []
        global_filter_idx = 0

        for layer_idx, layer_acts in enumerate(phrase_acts_per_layer):
            kernel_size = self.model.kernel_sizes[layer_idx]
            F_dim = layer_acts.shape[1]

            for local_f in range(F_dim):
                # Pooled activations for this filter
                pooled_f = pooled[:, global_filter_idx]
                act_range = float(pooled_f.max() - pooled_f.min())
                is_active = act_range >= self.activation_threshold

                finfo = FilterInfo(
                    filter_idx=global_filter_idx,
                    conv_layer_idx=layer_idx,
                    local_filter_idx=local_f,
                    kernel_size=kernel_size,
                    output_weight=float(output_weights[global_filter_idx]),
                    activation_range=act_range,
                    is_active=is_active,
                )

                if is_active:
                    # Extract top phrases
                    finfo.top_phrases = self._extract_top_phrases(
                        layer_acts[:, local_f, :],  # (N, P)
                        tokens, texts, kernel_size,
                    )

                filters.append(finfo)
                global_filter_idx += 1

        # --- Estimate treatment effects via OLS ---
        effect_estimates = None
        effect_cis = None
        r_squared_adj = None

        if estimate_effects:
            active_mask = np.array([f.is_active for f in filters])
            if active_mask.sum() > 0:
                active_pooled = pooled[:, active_mask]
                effect_estimates_full = np.full(len(filters), np.nan)
                effect_cis_full = np.full((len(filters), 2), np.nan)

                betas, cis, r2adj = _ols_with_bootstrap(
                    active_pooled, labels, n_bootstrap=n_bootstrap
                )

                active_indices = np.where(active_mask)[0]
                for i, aidx in enumerate(active_indices):
                    effect_estimates_full[aidx] = betas[i]
                    effect_cis_full[aidx] = cis[i]
                    filters[aidx].effect_estimate = float(betas[i])
                    filters[aidx].effect_ci = (float(cis[i, 0]), float(cis[i, 1]))

                effect_estimates = effect_estimates_full
                effect_cis = effect_cis_full
                r_squared_adj = r2adj

        # Sort filters by absolute output weight (descending)
        filters.sort(key=lambda f: abs(f.output_weight), reverse=True)

        return InterpretationResult(
            filters=filters,
            output_weights=output_weights,
            pooled_activations=pooled,
            effect_estimates=effect_estimates,
            effect_cis=effect_cis,
            r_squared_adj=r_squared_adj,
        )

    def _extract_top_phrases(
        self,
        filter_acts: np.ndarray,
        tokens: List[List[str]],
        texts: Optional[List[str]],
        kernel_size: int,
    ) -> List[dict]:
        """
        Extract the top-K most activated phrases for a single filter.

        Args:
            filter_acts: (N, P) activations for this filter across all
                phrases of all samples.
            tokens: Token strings per sample.
            texts: Original text strings per sample.
            kernel_size: K, the number of tokens per phrase.

        Returns:
            List of dicts describing the top phrases.
        """
        N, P = filter_acts.shape
        top_phrases = []
        seen_texts = set()

        # Flatten and find top activations
        flat = filter_acts.ravel()  # (N*P,)
        sorted_indices = np.argsort(flat)[::-1]

        for idx in sorted_indices:
            if len(top_phrases) >= self.top_k_phrases:
                break

            sample_idx = idx // P
            phrase_pos = idx % P

            # Extract token span
            start_tok = phrase_pos
            end_tok = start_tok + kernel_size
            if end_tok > len(tokens[sample_idx]):
                continue

            phrase_tokens = tokens[sample_idx][start_tok:end_tok]
            phrase_text = " ".join(
                t for t in phrase_tokens
                if t not in ("[PAD]", "<pad>")
            )

            # Deduplicate
            if self.deduplicate_phrases and phrase_text in seen_texts:
                continue
            seen_texts.add(phrase_text)

            phrase_info = {
                'text': phrase_text,
                'tokens': phrase_tokens,
                'activation': float(flat[idx]),
                'sample_idx': int(sample_idx),
                'phrase_position': int(phrase_pos),
                'sample_text': texts[sample_idx] if texts is not None else None,
            }
            top_phrases.append(phrase_info)

        return top_phrases

    def summary_table(
        self,
        result: InterpretationResult,
        only_active: bool = True,
        max_phrases: int = 3,
    ) -> List[dict]:
        """
        Produce a summary table similar to Tables 1 and 2 in the paper.

        Args:
            result: InterpretationResult from interpret().
            only_active: Only include active filters.
            max_phrases: Max phrases to show per filter.

        Returns:
            List of dicts, one per filter, with keys:
                'filter', 'W_out', 'beta', 'CI', 'top_phrases', 'label'
        """
        rows = []
        for f in result.filters:
            if only_active and not f.is_active:
                continue

            phrase_strs = [
                f'"{p["text"]}"' for p in f.top_phrases[:max_phrases]
            ]
            ci_str = (
                f"[{f.effect_ci[0]:.2f}, {f.effect_ci[1]:.2f}]"
                if f.effect_ci else "N/A"
            )

            rows.append({
                'filter': f.filter_idx,
                'kernel_size': f.kernel_size,
                'W_out': f"{f.output_weight:.2f}",
                'beta': f"{f.effect_estimate:.2f}" if f.effect_estimate is not None else "N/A",
                'CI': ci_str,
                'top_phrases': ", ".join(phrase_strs),
                'label': f.manual_label or "",
                'activation_range': f"{f.activation_range:.3f}",
            })

        return rows

    def filter_correlation_matrix(
        self,
        result: InterpretationResult,
        only_active: bool = True,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute pairwise Pearson correlations between filter pooled activations.

        Returns:
            corr_matrix: (K, K) correlation matrix.
            filter_indices: List of filter indices included.
        """
        if only_active:
            active = [f for f in result.filters if f.is_active]
            indices = [f.filter_idx for f in active]
        else:
            indices = list(range(result.pooled_activations.shape[1]))

        acts = result.pooled_activations[:, indices]
        corr = np.corrcoef(acts.T)
        return corr, indices


def _ols_with_bootstrap(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    OLS regression of y on X with bootstrap confidence intervals.

    Per the paper, treatment effects are estimated by regressing the
    outcome labels on max-pooled filter activations in the test set.

    Args:
        X: (N, K) matrix of active filter pooled activations.
        y: (N,) binary outcome labels.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level for CIs.

    Returns:
        betas: (K,) OLS coefficient estimates from full sample.
        cis: (K, 2) bootstrap percentile confidence intervals.
        r2_adj: Adjusted R-squared from the full-sample regression.
    """
    N, K = X.shape
    y = y.ravel()

    # Full-sample OLS
    X_with_intercept = np.column_stack([np.ones(N), X])
    try:
        betas_full = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        logger.warning("OLS failed; returning zeros.")
        return np.zeros(K), np.zeros((K, 2)), 0.0

    betas = betas_full[1:]  # exclude intercept

    # R-squared adjusted
    y_pred = X_with_intercept @ betas_full
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    r2_adj = 1 - (1 - r2) * (N - 1) / max(N - K - 1, 1)

    # Bootstrap for CIs
    boot_betas = np.zeros((n_bootstrap, K))
    rng = np.random.default_rng(42)
    for b in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        X_b = X_with_intercept[idx]
        y_b = y[idx]
        try:
            betas_b = np.linalg.lstsq(X_b, y_b, rcond=None)[0]
            boot_betas[b] = betas_b[1:]
        except np.linalg.LinAlgError:
            boot_betas[b] = betas

    lower = alpha / 2 * 100
    upper = (1 - alpha / 2) * 100
    cis = np.column_stack([
        np.percentile(boot_betas, lower, axis=0),
        np.percentile(boot_betas, upper, axis=0),
    ])

    return betas, cis, r2_adj


def compute_mse(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """
    Fit OLS on (X, y) and compute MSE on (X_test, y_test).

    Args:
        X, y: Training features and labels.
        X_test, y_test: Test features and labels.

    Returns:
        Mean squared error on test set.
    """
    N = X.shape[0]
    X_aug = np.column_stack([np.ones(N), X])
    betas = np.linalg.lstsq(X_aug, y.ravel(), rcond=None)[0]

    N_test = X_test.shape[0]
    X_test_aug = np.column_stack([np.ones(N_test), X_test])
    y_pred = X_test_aug @ betas

    return float(np.mean((y_test.ravel() - y_pred) ** 2))
