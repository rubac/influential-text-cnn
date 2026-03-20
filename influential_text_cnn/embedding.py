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

    For texts longer than the BERT context window (typically 512 tokens),
    the text is split into overlapping chunks, each chunk is embedded
    separately, and the chunk embeddings are concatenated along the
    sequence dimension.  The CNN's max-pooling then selects the best-
    matching phrase from any position in the full document.

    Args:
        model_name: HuggingFace model name/path.
        max_tokens: Total output sequence length U.  All documents are
            padded/truncated to exactly this many token positions.
            For long documents, set this to a large value (e.g. 14000).
        device: Torch device string.
        batch_size: Batch size for BERT inference (per-chunk).
        chunk_size: Maximum tokens per BERT chunk (≤ 510 to leave room
            for [CLS] and [SEP]).  Defaults to 510.
        chunk_overlap: Number of overlapping tokens between consecutive
            chunks.  Overlap ensures phrases that span chunk boundaries
            are captured at least once.  Defaults to 50.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_tokens: int = 150,
        device: str = "cpu",
        batch_size: int = 32,
        chunk_size: int = 510,
        chunk_overlap: int = 50,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = min(chunk_size, 510)  # BERT hard limit
        self.chunk_overlap = chunk_overlap
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

    def _needs_chunking(self, texts: List[str]) -> bool:
        """Check whether any text exceeds a single BERT window."""
        return self.max_tokens > self.chunk_size

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Compute BERT last-hidden-state embeddings for a list of texts.

        Short texts (max_tokens ≤ chunk_size) are embedded in a single
        pass.  Long texts are chunked, embedded per-chunk, and the
        content token embeddings are concatenated.

        Args:
            texts: List of N text strings.

        Returns:
            EmbeddingResult with embeddings (N, max_tokens, D) and tokens.
        """
        self._load_model()

        if self._needs_chunking(texts):
            return self._embed_chunked(texts)
        else:
            return self._embed_simple(texts)

    # ------------------------------------------------------------------
    # Simple path (short texts, no chunking)
    # ------------------------------------------------------------------

    def _embed_simple(self, texts: List[str]) -> EmbeddingResult:
        """Embed texts that fit in a single BERT window."""
        all_embeddings = []
        all_tokens = []

        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(texts))
            batch_texts = texts[start:end]

            encoded = self._tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )

            for text in batch_texts:
                toks = self._tokenizer.tokenize(text)[: self.max_tokens - 2]
                toks = (
                    [self._tokenizer.cls_token]
                    + toks
                    + [self._tokenizer.sep_token]
                )
                toks = toks + [self._tokenizer.pad_token] * (
                    self.max_tokens - len(toks)
                )
                all_tokens.append(toks[: self.max_tokens])

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

    # ------------------------------------------------------------------
    # Chunked path (long texts)
    # ------------------------------------------------------------------

    def _embed_chunked(self, texts: List[str]) -> EmbeddingResult:
        """
        Embed long texts by chunking.

        Strategy:
        1. Tokenize each text fully (no truncation).
        2. Split the token list into overlapping chunks of chunk_size.
        3. Embed each chunk with BERT (adding [CLS]/[SEP] around it).
        4. Keep only the content-token embeddings (strip [CLS]/[SEP]).
        5. For overlapping regions, keep the embeddings from the chunk
           where the token is further from the boundary (better context).
        6. Concatenate chunks and pad/truncate to max_tokens.
        """
        D = self._model.config.hidden_size
        stride = self.chunk_size - self.chunk_overlap
        pad_token = self._tokenizer.pad_token or "[PAD]"

        all_embeddings = []
        all_tokens = []

        for text_idx, text in enumerate(texts):
            # Full tokenization (no truncation)
            full_tokens = self._tokenizer.tokenize(text)

            if len(full_tokens) == 0:
                # Empty text
                emb = np.zeros((self.max_tokens, D), dtype=np.float32)
                toks = [pad_token] * self.max_tokens
                all_embeddings.append(emb)
                all_tokens.append(toks)
                continue

            # Create chunks with overlap
            chunks = []
            for chunk_start in range(0, len(full_tokens), stride):
                chunk_end = min(chunk_start + self.chunk_size, len(full_tokens))
                chunk_toks = full_tokens[chunk_start:chunk_end]
                chunks.append((chunk_start, chunk_end, chunk_toks))
                if chunk_end >= len(full_tokens):
                    break

            # Embed all chunks for this text
            chunk_embeddings = []
            for _, _, chunk_toks in chunks:
                chunk_emb = self._embed_single_chunk(chunk_toks)
                chunk_embeddings.append(chunk_emb)  # (len(chunk_toks), D)

            # Resolve overlaps: for each token position in the original
            # sequence, pick the embedding from the chunk where the token
            # is most central (furthest from edges → better BERT context).
            n_real_tokens = len(full_tokens)
            merged = np.zeros((n_real_tokens, D), dtype=np.float32)
            best_dist = np.full(n_real_tokens, -1.0)

            for (chunk_start, chunk_end, chunk_toks), chunk_emb in zip(
                chunks, chunk_embeddings
            ):
                chunk_len = chunk_end - chunk_start
                for local_pos in range(chunk_len):
                    global_pos = chunk_start + local_pos
                    # Distance from nearest chunk edge
                    dist = min(local_pos, chunk_len - 1 - local_pos)
                    if dist > best_dist[global_pos]:
                        best_dist[global_pos] = dist
                        merged[global_pos] = chunk_emb[local_pos]

            # Truncate or pad to max_tokens
            if n_real_tokens >= self.max_tokens:
                emb = merged[: self.max_tokens]
                toks = full_tokens[: self.max_tokens]
            else:
                pad_len = self.max_tokens - n_real_tokens
                emb = np.concatenate([
                    merged,
                    np.zeros((pad_len, D), dtype=np.float32),
                ], axis=0)
                toks = full_tokens + [pad_token] * pad_len

            all_embeddings.append(emb)
            all_tokens.append(toks)

            if (text_idx + 1) % 50 == 0 or (text_idx + 1) == len(texts):
                n_chunks_total = len(chunks)
                logger.info(
                    f"  Embedded {text_idx + 1}/{len(texts)} texts "
                    f"({(text_idx + 1) / len(texts):.0%}) — "
                    f"last text: {n_real_tokens} tokens, "
                    f"{n_chunks_total} chunks"
                )

        embeddings = np.stack(all_embeddings, axis=0)  # (N, max_tokens, D)
        return EmbeddingResult(embeddings=embeddings, tokens=all_tokens)

    def _embed_single_chunk(self, chunk_tokens: List[str]) -> np.ndarray:
        """
        Embed a single chunk of tokens through BERT.

        Wraps the tokens with [CLS] and [SEP], runs BERT, and returns
        only the content-token embeddings (stripping [CLS]/[SEP]).

        Args:
            chunk_tokens: List of wordpiece tokens for this chunk.

        Returns:
            (len(chunk_tokens), D) numpy array of embeddings.
        """
        # Convert tokens to IDs, adding special tokens
        ids = self._tokenizer.convert_tokens_to_ids(chunk_tokens)
        cls_id = self._tokenizer.cls_token_id
        sep_id = self._tokenizer.sep_token_id

        input_ids = [cls_id] + ids + [sep_id]
        attention_mask = [1] * len(input_ids)

        input_ids_t = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask_t = torch.tensor(
            [attention_mask], dtype=torch.long
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
            )
            # Shape: (1, seq_len, D) → strip [CLS] and [SEP]
            hidden = outputs.last_hidden_state[0]  # (seq_len, D)
            content = hidden[1:-1]  # strip [CLS] at 0, [SEP] at -1

        return content.cpu().numpy()  # (len(chunk_tokens), D)


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
