"""
High-level pipeline that orchestrates the full workflow.

Per the paper's experimental setup (Sections 3-5):
1. Embed texts using BERT
2. Split into train/test sets
3. Tune hyperparameters using k-fold CV on training set
4. Re-train final model on full training set (with 20% val split)
5. Interpret filters on test set
6. Estimate treatment effects on test set
7. Compare to benchmark methods
"""

import numpy as np
import torch
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field

from .model import InfluentialTextCNN, InfluentialTextLoss
from .embedding import BERTEmbedder, PrecomputedEmbedder, EmbeddingResult
from .training import Trainer, TrainingHistory
from .interpretation import FilterInterpreter, InterpretationResult, compute_mse
from .tuning import tune_hyperparameters, TuningReport
from .benchmarks import RegularizedLogisticRegression, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete results from the pipeline."""
    # Model and training
    model: InfluentialTextCNN = None
    training_history: TrainingHistory = None
    test_metrics: Dict[str, float] = None

    # Interpretation
    interpretation: InterpretationResult = None

    # Tuning
    tuning_report: TuningReport = None

    # Embeddings
    embedding_result: EmbeddingResult = None
    train_indices: np.ndarray = None
    test_indices: np.ndarray = None

    # Benchmark comparisons
    benchmark_results: Dict[str, BenchmarkResult] = field(default_factory=dict)

    # Evaluation metrics comparison
    cnn_r2_adj: float = None
    cnn_mse: float = None


class InfluentialTextPipeline:
    """
    End-to-end pipeline for discovering influential text treatments.

    Example usage:

        pipeline = InfluentialTextPipeline(
            model_name="google/bert_uncased_L-2_H-128_A-2",
            max_tokens=250,
        )
        result = pipeline.run(
            texts=texts,
            labels=labels,
            tune=True,
        )

        # Print summary table
        table = pipeline.summary_table(result)
        for row in table:
            print(row)

    Args:
        model_name: HuggingFace BERT model name for embeddings.
        max_tokens: Maximum tokens per text sample (U).
        test_fraction: Fraction of data for test set.
        device: Device string.
        random_seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model_name: str = "google/bert_uncased_L-2_H-128_A-2",
        max_tokens: int = 250,
        test_fraction: float = 0.2,
        device: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.test_fraction = test_fraction
        self.random_seed = random_seed

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def run(
        self,
        texts: List[str],
        labels: np.ndarray,
        # Embedding options
        precomputed_embeddings: Optional[np.ndarray] = None,
        precomputed_tokens: Optional[List[List[str]]] = None,
        # Tuning options
        tune: bool = True,
        param_grid: Optional[Dict[str, List]] = None,
        n_folds: int = 5,
        # Model options (used if tune=False)
        num_filters: int = 8,
        kernel_sizes: Optional[List[int]] = None,
        lambda_conv_ker: float = 0.001,
        lambda_conv_act: float = 3.0,
        lambda_out_ker: float = 0.0001,
        learning_rate: float = 0.0001,
        # Training options
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 15,
        # Interpretation options
        estimate_effects: bool = True,
        n_bootstrap: int = 1000,
        # Benchmark options
        run_benchmarks: bool = True,
        ngram_range: Tuple[int, int] = (3, 3),
        min_ngram_frequency: int = 50,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            texts: List of N text strings.
            labels: (N,) binary outcome labels.
            precomputed_embeddings: Optional (N, U, D) array to skip BERT.
            precomputed_tokens: Optional list of token lists (if using
                precomputed embeddings).
            tune: Whether to perform hyperparameter tuning.
            param_grid: Custom parameter grid for tuning.
            n_folds: Number of CV folds for tuning.
            num_filters: Filters per conv layer (if tune=False).
            kernel_sizes: Kernel sizes (if tune=False).
            lambda_conv_ker, lambda_conv_act, lambda_out_ker: Regularization
                strengths (if tune=False).
            learning_rate: Learning rate (if tune=False).
            epochs, batch_size, patience: Training parameters.
            estimate_effects: Whether to estimate treatment effects.
            n_bootstrap: Bootstrap samples for effect CIs.
            run_benchmarks: Whether to run benchmark comparisons.
            ngram_range: N-gram range for RLR benchmark.
            min_ngram_frequency: Min frequency for RLR benchmark.

        Returns:
            PipelineResult with all outputs.
        """
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        result = PipelineResult()

        labels = np.asarray(labels).ravel()
        N = len(texts)

        # ===== Step 1: Embed texts =====
        logger.info("Step 1: Computing embeddings...")
        if precomputed_embeddings is not None:
            embedder = PrecomputedEmbedder(
                precomputed_embeddings, precomputed_tokens
            )
        else:
            embedder = BERTEmbedder(
                model_name=self.model_name,
                max_tokens=self.max_tokens,
                device=self.device,
            )

        emb_result = embedder.embed(texts)
        result.embedding_result = emb_result
        embedding_dim = emb_result.embeddings.shape[2]

        # ===== Step 2: Train/test split =====
        logger.info("Step 2: Creating train/test split...")
        perm = np.random.permutation(N)
        n_test = int(N * self.test_fraction)
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        result.train_indices = train_idx
        result.test_indices = test_idx

        train_emb = emb_result.embeddings[train_idx]
        test_emb = emb_result.embeddings[test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        train_tokens = [emb_result.tokens[i] for i in train_idx]
        test_tokens = [emb_result.tokens[i] for i in test_idx]
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        logger.info(
            f"  Train: {len(train_idx)} samples "
            f"({train_labels.mean():.2%} positive)"
        )
        logger.info(
            f"  Test: {len(test_idx)} samples "
            f"({test_labels.mean():.2%} positive)"
        )

        # ===== Step 3: Hyperparameter tuning (optional) =====
        if tune:
            logger.info("Step 3: Tuning hyperparameters...")
            tuning_report = tune_hyperparameters(
                embeddings=train_emb,
                labels=train_labels,
                param_grid=param_grid,
                n_folds=n_folds,
                epochs=epochs,
                batch_size=batch_size,
                patience=patience,
                device=self.device,
            )
            result.tuning_report = tuning_report
            best = tuning_report.best_params

            num_filters = best.get('num_filters', num_filters)
            kernel_sizes = best.get('kernel_sizes', kernel_sizes or [5])
            lambda_conv_ker = best.get('lambda_conv_ker', lambda_conv_ker)
            lambda_conv_act = best.get('lambda_conv_act', lambda_conv_act)
            lambda_out_ker = best.get('lambda_out_ker', lambda_out_ker)
            learning_rate = best.get('learning_rate', learning_rate)

            logger.info(f"  Best params: {best}")
        else:
            if kernel_sizes is None:
                kernel_sizes = [5]

        # ===== Step 4: Train final model =====
        logger.info("Step 4: Training final model on full training set...")
        model = InfluentialTextCNN(
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_sizes=kernel_sizes,
        )
        loss_fn = InfluentialTextLoss(
            model=model,
            lambda_conv_ker=lambda_conv_ker,
            lambda_conv_act=lambda_conv_act,
            lambda_out_ker=lambda_out_ker,
        )
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=self.device,
        )

        history = trainer.fit(train_emb, train_labels)
        result.model = model
        result.training_history = history

        # ===== Step 5: Evaluate on test set =====
        logger.info("Step 5: Evaluating on test set...")
        test_metrics = trainer.evaluate(test_emb, test_labels)
        result.test_metrics = test_metrics
        logger.info(
            f"  Test accuracy: {test_metrics['accuracy']:.3f} | "
            f"F1: {test_metrics['f1']:.3f}"
        )

        # ===== Step 6: Interpret filters =====
        logger.info("Step 6: Interpreting filters...")
        interpreter = FilterInterpreter(model)
        interpretation = interpreter.interpret(
            embeddings=test_emb,
            tokens=test_tokens,
            labels=test_labels,
            texts=test_texts,
            n_bootstrap=n_bootstrap,
            estimate_effects=estimate_effects,
            device=self.device,
        )
        result.interpretation = interpretation

        if interpretation.r_squared_adj is not None:
            result.cnn_r2_adj = interpretation.r_squared_adj
            logger.info(f"  R²_adj (CNN): {interpretation.r_squared_adj:.4f}")

        # Compute MSE
        active_mask = np.array([
            f.is_active for f in interpretation.filters
        ])
        if active_mask.sum() > 0:
            # Get pooled activations for train set too
            train_pooled = _get_pooled(model, train_emb, self.device)
            test_pooled = interpretation.pooled_activations

            train_active = train_pooled[:, active_mask]
            test_active = test_pooled[:, active_mask]

            cnn_mse = compute_mse(train_active, train_labels, test_active, test_labels)
            result.cnn_mse = cnn_mse
            logger.info(f"  MSE (CNN): {cnn_mse:.4f}")

        # ===== Step 7: Benchmark comparisons =====
        if run_benchmarks:
            logger.info("Step 7: Running benchmark comparisons...")
            total_filters = model.total_filters

            try:
                rlr = RegularizedLogisticRegression(
                    ngram_range=ngram_range,
                    max_features=total_filters,
                    min_frequency=min_ngram_frequency,
                )
                rlr.fit(train_texts, train_labels)
                rlr_result = rlr.evaluate(
                    train_texts, train_labels, test_texts, test_labels
                )
                result.benchmark_results['RLR'] = rlr_result
                logger.info(
                    f"  RLR - R²_adj: {rlr_result.r_squared_adj:.4f} | "
                    f"MSE: {rlr_result.mse:.4f}"
                )
            except Exception as e:
                logger.warning(f"RLR benchmark failed: {e}")

        logger.info("Pipeline complete!")
        return result

    @staticmethod
    def summary_table(result: PipelineResult, max_phrases: int = 3) -> List[dict]:
        """Generate summary table from pipeline results."""
        if result.interpretation is None:
            return []

        interpreter = FilterInterpreter(result.model)
        return interpreter.summary_table(
            result.interpretation,
            max_phrases=max_phrases,
        )

    @staticmethod
    def print_summary(result: PipelineResult, max_phrases: int = 3):
        """Print a formatted summary of results."""
        print("\n" + "=" * 80)
        print("INFLUENTIAL TEXT CNN - RESULTS SUMMARY")
        print("=" * 80)

        if result.test_metrics:
            print(f"\nTest Set Performance:")
            for k, v in result.test_metrics.items():
                print(f"  {k}: {v:.4f}")

        if result.cnn_r2_adj is not None:
            print(f"\nModel Fit (OLS on test set):")
            print(f"  R²_adj: {result.cnn_r2_adj:.4f}")
            if result.cnn_mse is not None:
                print(f"  MSE: {result.cnn_mse:.4f}")

        if result.benchmark_results:
            print(f"\nBenchmark Comparisons:")
            for name, br in result.benchmark_results.items():
                print(f"  {name}: R²_adj={br.r_squared_adj:.4f}, MSE={br.mse:.4f}")
                for feat in br.features[:5]:
                    print(f"    {feat.label}: {feat.coefficient:.4f}")

        if result.interpretation:
            print(f"\nDiscovered Text Treatments:")
            print("-" * 80)

            table = InfluentialTextPipeline.summary_table(result, max_phrases)
            for row in table:
                print(
                    f"Filter {row['filter']:2d} "
                    f"(K={row['kernel_size']}) | "
                    f"W_out={row['W_out']:>6s} | "
                    f"β={row['beta']:>6s} | "
                    f"CI={row['CI']}"
                )
                print(f"  Phrases: {row['top_phrases']}")
                if row['label']:
                    print(f"  Label: {row['label']}")
                print()

        print("=" * 80)


def _get_pooled(
    model: InfluentialTextCNN,
    embeddings: np.ndarray,
    device: str,
) -> np.ndarray:
    """Get pooled activations for a dataset."""
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
    return np.concatenate(all_pooled, axis=0)
