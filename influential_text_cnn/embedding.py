"""
Embedding module for the Influential Text CNN pipeline.

Provides BERT-based text embedding and a precomputed-embedding wrapper.
"""

import numpy as np
import torch
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Container for embedding outputs."""
    embeddings: np.ndarray           # (N, U, D)
    tokens: List[List[str]]          # tokenized text per sample


class BERTEmbedder:
    """
    Computes contextualized embeddings using a pre-trained BERT model.

    Args:
        model_name: HuggingFace model name/path.
        max_tokens: Maximum number of tokens per sample (U).
        device: Torch device string.
        batch_size: Batch size for embedding computation.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 150,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device
        self.batch_size = batch_size
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer, AutoModel

            logger.info(f"Loading BERT model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info(
                f"  Model loaded. Embedding dim: {self._model.config.hidden_size}, "
                f"Device: {self.device}"
            )

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._model.config.hidden_size

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Compute BERT last-hidden-state embeddings for a list of texts.

        Args:
            texts: List of N text strings.

        Returns:
            EmbeddingResult with embeddings (N, max_tokens, D) and tokens.
        """
        self._load_model()

        all_embeddings = []
        all_tokens = []

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:end]

            # Tokenize
            encoded = self._tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )

            # Store token strings for interpretation
            for text in batch_texts:
                toks = self._tokenizer.tokenize(text)[: self.max_tokens - 2]
                toks = (
                    [self._tokenizer.cls_token]
                    + toks
                    + [self._tokenizer.sep_token]
                )
                # Pad to max_tokens
                toks = toks + [self._tokenizer.pad_token] * (
                    self.max_tokens - len(toks)
                )
                all_tokens.append(toks[: self.max_tokens])

            # Compute embeddings
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self._model(**inputs)
                emb = outputs.last_hidden_state.cpu().numpy()
                all_embeddings.append(emb)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                logger.info(
                    f"  Embedded {end}/{len(texts)} texts "
                    f"({end / len(texts):.0%})"
                )

        embeddings = np.concatenate(all_embeddings, axis=0)
        return EmbeddingResult(embeddings=embeddings, tokens=all_tokens)


class PrecomputedEmbedder:
    """
    Wrapper for pre-computed embeddings (skips BERT computation).

    Useful when you've already computed embeddings separately.

    Args:
        embeddings: (N, U, D) array of pre-computed embeddings.
        tokens: Optional list of token string lists per sample.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        tokens: Optional[List[List[str]]] = None,
    ):
        self._embeddings = embeddings
        self._tokens = tokens

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """Return the pre-computed embeddings."""
        N = self._embeddings.shape[0]
        if len(texts) != N:
            raise ValueError(
                f"Number of texts ({len(texts)}) does not match "
                f"number of embeddings ({N})"
            )

        tokens = self._tokens
        if tokens is None:
            # Create placeholder tokens
            U = self._embeddings.shape[1]
            tokens = [
                [f"tok_{j}" for j in range(U)]
                for _ in range(N)
            ]

        logger.info(
            f"  Using precomputed embeddings: shape {self._embeddings.shape}"
        )
        return EmbeddingResult(
            embeddings=self._embeddings,
            tokens=tokens,
        )
