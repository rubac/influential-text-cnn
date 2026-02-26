"""
Benchmark methods for comparison (Section 4.5, Appendix B).

1. Regularized Logistic Regression (RLR) on n-gram features.
   - L1-penalized logistic regression on n-gram counts.
   - Selects up to K n-grams that are predictive of the outcome.
   - Clear interpretation: selected n-grams are the text features.

2. Fong & Grimmer (F&G) method is implemented in R (texteffect package).
   We provide a wrapper that calls the R implementation if available,
   but it is NOT required.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import Counter
from dataclasses import dataclass, field
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkFeature:
    """A text feature identified by a benchmark method."""
    label: str
    coefficient: float
    feature_type: str = "ngram"  # "ngram" or "topic"


@dataclass
class BenchmarkResult:
    """Result from a benchmark method."""
    method: str
    features: List[BenchmarkFeature]
    r_squared_adj: Optional[float] = None
    mse: Optional[float] = None
    feature_matrix_train: Optional[np.ndarray] = None
    feature_matrix_test: Optional[np.ndarray] = None


class RegularizedLogisticRegression:
    """
    Regularized logistic regression on n-gram features.

    Per Appendix B: "We perform regularized logistic regression with a L1
    penalty on 3-grams in each corpus... We chose the penalty parameter to
    be the minimum magnitude such that at most 16 3-grams were selected."

    Args:
        ngram_range: Tuple (min_n, max_n) for n-gram extraction.
        max_features: Maximum n-grams to select (match num_filters).
        min_frequency: Minimum document frequency for n-grams.
        stop_words: List of stop words to exclude, or 'english'.
        max_vocab_size: Maximum vocabulary size before regularization.
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (3, 3),
        max_features: int = 16,
        min_frequency: int = 50,
        stop_words: Optional[str] = "english",
        max_vocab_size: int = 10000,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.min_frequency = min_frequency
        self.stop_words = stop_words
        self.max_vocab_size = max_vocab_size

        self._vectorizer = None
        self._model = None
        self._selected_ngrams = None
        self._selected_indices = None

    def fit(
        self,
        texts: List[str],
        labels: np.ndarray,
    ) -> "RegularizedLogisticRegression":
        """
        Fit the RLR model on training data.

        Finds the minimum L1 penalty such that at most max_features
        n-grams are selected.

        Args:
            texts: List of N training text strings.
            labels: (N,) binary labels.

        Returns:
            self
        """
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression

        # Build n-gram vocabulary
        self._vectorizer = CountVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_frequency,
            max_features=self.max_vocab_size,
            stop_words=self.stop_words if self.stop_words != "none" else None,
        )
        X = self._vectorizer.fit_transform(texts)
        feature_names = self._vectorizer.get_feature_names_out()

        logger.info(f"Vocabulary size: {len(feature_names)} {self.ngram_range}-grams")

        # Binary search for the right C (inverse penalty) to select
        # at most max_features non-zero coefficients
        C_low, C_high = 1e-5, 100.0
        best_model = None
        best_C = C_low

        for _ in range(50):  # binary search iterations
            C_mid = (C_low + C_high) / 2
            model = LogisticRegression(
                penalty='l1',
                C=C_mid,
                solver='liblinear',
                max_iter=1000,
                random_state=42,
            )
            model.fit(X, labels.ravel())
            n_selected = np.sum(model.coef_[0] != 0)

            if n_selected <= self.max_features:
                best_model = model
                best_C = C_mid
                C_low = C_mid  # try less regularization
            else:
                C_high = C_mid  # try more regularization

            if abs(C_high - C_low) < 1e-8:
                break

        if best_model is None:
            # Fallback: use very strong regularization
            best_model = LogisticRegression(
                penalty='l1', C=1e-5, solver='liblinear',
                max_iter=1000, random_state=42,
            )
            best_model.fit(X, labels.ravel())

        self._model = best_model
        coef = best_model.coef_[0]
        nonzero = np.where(coef != 0)[0]

        # Sort by absolute coefficient
        sorted_idx = nonzero[np.argsort(np.abs(coef[nonzero]))[::-1]]
        self._selected_indices = sorted_idx[:self.max_features]
        self._selected_ngrams = [
            (feature_names[i], float(coef[i]))
            for i in self._selected_indices
        ]

        logger.info(
            f"Selected {len(self._selected_ngrams)} n-grams "
            f"with C={best_C:.6f}"
        )

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to the selected n-gram feature matrix.

        Args:
            texts: List of text strings.

        Returns:
            (N, K) matrix of selected n-gram counts.
        """
        X = self._vectorizer.transform(texts)
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            return X[:, self._selected_indices].toarray()
        return X.toarray()

    def get_features(self) -> List[BenchmarkFeature]:
        """Get the selected n-gram features with coefficients."""
        features = []
        if self._selected_ngrams:
            for ngram, coef in self._selected_ngrams:
                features.append(BenchmarkFeature(
                    label=ngram,
                    coefficient=coef,
                    feature_type="ngram",
                ))
        return features

    def evaluate(
        self,
        train_texts: List[str],
        train_labels: np.ndarray,
        test_texts: List[str],
        test_labels: np.ndarray,
    ) -> BenchmarkResult:
        """
        Evaluate: compute R²_adj and MSE using OLS on selected features.

        Per the paper, methods are compared by:
        1. R²_adj of linear regression of outcome on text treatments
        2. MSE of that regression on out-of-sample data

        Args:
            train_texts, train_labels: Training data.
            test_texts, test_labels: Test data.

        Returns:
            BenchmarkResult with metrics and features.
        """
        X_train = self.transform(train_texts)
        X_test = self.transform(test_texts)
        y_train = train_labels.ravel()
        y_test = test_labels.ravel()

        # OLS on training set
        N, K = X_train.shape
        X_aug = np.column_stack([np.ones(N), X_train])
        betas = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]

        # R²_adj on test set
        N_test = X_test.shape[0]
        X_test_aug = np.column_stack([np.ones(N_test), X_test])
        y_pred = X_test_aug @ betas

        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - y_test.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        r2_adj = 1 - (1 - r2) * (N_test - 1) / max(N_test - K - 1, 1)

        mse = float(np.mean((y_test - y_pred) ** 2))

        return BenchmarkResult(
            method="RLR",
            features=self.get_features(),
            r_squared_adj=r2_adj,
            mse=mse,
            feature_matrix_train=X_train,
            feature_matrix_test=X_test,
        )
