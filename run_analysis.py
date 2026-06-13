"""
=============================================================================
Influential Text CNN — Run the analysis on your own data
=============================================================================

A no-coding-required entry point: edit the CONFIGURATION section below
(Steps 1-3), then run

    python run_analysis.py

Results are written to the results/ folder:
    - filter_interpretations.csv  one row per discovered text pattern
    - metrics.json                fit statistics and benchmark comparison
    - model.pt                    the trained model weights

See README.md and GETTING_STARTED.md for a full walk-through.
=============================================================================
"""

import sys
import os
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Make the package importable when running this script from its own folder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from influential_text_cnn import InfluentialTextPipeline


# =============================================================================
# STEP 1: YOUR DATA
# =============================================================================
# Point these at your file and the two columns you care about.

DATA_FILE       = "your_data.csv"     # <-- Path to your CSV / TSV / XLSX
TEXT_COLUMN     = "text"              # <-- Column holding the text
OUTCOME_COLUMN  = "outcome"          # <-- Column holding the outcome

# =============================================================================
# STEP 2: BERT MODEL (provides the word embeddings)
# =============================================================================
# English:  "prajjwal1/bert-tiny" (fast) | "distilbert-base-uncased" | "bert-base-uncased"
# Chinese:  "bert-base-chinese" | "hfl/chinese-roberta-wwm-ext"
# Multi:    "bert-base-multilingual-cased"
#
# Start with bert-tiny to confirm everything runs, then switch to a larger
# model for your final results.

BERT_MODEL = "prajjwal1/bert-tiny"

# =============================================================================
# STEP 3: TASK + MODEL PARAMETERS  (defaults follow the paper)
# =============================================================================

# Task type:
#   "binary"      — outcome is 0/1   (uses BCE loss, reports accuracy/F1)
#   "continuous"  — outcome is numeric (uses MSE loss, reports R²/RMSE)
TASK = "binary"

MAX_LENGTH      = 150          # Max tokens kept per text
KERNEL_SIZES    = [5, 7]       # Phrase lengths to scan for (in tokens)
NUM_FILTERS     = 8            # Patterns (filters) learned per kernel size
EPOCHS          = 100          # Max training epochs
PATIENCE        = 15           # Early-stopping patience
LEARNING_RATE   = 0.001
LAMBDA_CONV_KER = 0.001        # L2 on conv weights (smoothness)
LAMBDA_CONV_ACT = 3.0          # Filter-diversity penalty (higher = more distinct patterns)
LAMBDA_OUT_KER  = 0.0001       # L1 on output weights (sparsity)

# Long-document support:
#   Texts longer than CHUNK_SIZE tokens are split into overlapping chunks,
#   each embedded with BERT, then stitched back together. To cover the full
#   document set MAX_LENGTH to roughly your longest text's token count
#   (e.g. 14000). This uses more memory but loses no text.
CHUNK_SIZE    = 510            # Tokens per BERT call (hard max 510)
CHUNK_OVERLAP = 50             # Overlap between consecutive chunks

# =============================================================================
# (No changes needed below this line)
# =============================================================================

def main():
    import pandas as pd

    # --- Load ---------------------------------------------------------------
    logger.info(f"Loading {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        sys.exit(
            f"ERROR: Data file '{DATA_FILE}' not found.\n"
            f"Edit DATA_FILE at the top of run_analysis.py to point at your CSV."
        )
    if DATA_FILE.endswith(".tsv"):
        df = pd.read_csv(DATA_FILE, sep="\t")
    elif DATA_FILE.endswith((".xls", ".xlsx")):
        df = pd.read_excel(DATA_FILE)
    else:
        df = pd.read_csv(DATA_FILE)

    for col in (TEXT_COLUMN, OUTCOME_COLUMN):
        if col not in df.columns:
            sys.exit(
                f"ERROR: Column '{col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

    texts  = df[TEXT_COLUMN].astype(str).tolist()
    labels = df[OUTCOME_COLUMN].values.astype(float)

    # Validate outcome
    if TASK == "binary":
        unique = set(np.unique(labels))
        if not unique.issubset({0.0, 1.0}):
            sys.exit(
                f"ERROR: For task='binary', the outcome must be 0/1. Found: {unique}\n"
                f"Set TASK = 'continuous' if your outcome is numeric."
            )

    # Drop empty texts
    valid = [len(t.strip()) > 0 for t in texts]
    if not all(valid):
        n_drop = sum(not v for v in valid)
        logger.info(f"  Dropping {n_drop} empty texts")
        texts  = [t for t, v in zip(texts, valid) if v]
        labels = labels[np.array(valid)]

    if TASK == "binary":
        logger.info(f"  {len(texts)} texts | {labels.mean():.1%} positive")
    else:
        logger.info(
            f"  {len(texts)} texts | outcome mean={labels.mean():.3f}, "
            f"sd={labels.std():.3f}"
        )

    # --- Pipeline -----------------------------------------------------------
    pipeline = InfluentialTextPipeline(
        model_name=BERT_MODEL,
        max_tokens=MAX_LENGTH,
        test_fraction=0.2,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    result = pipeline.run(
        texts=texts,
        labels=labels,
        task=TASK,
        tune=False,                    # set True to grid-search (slow; see README)
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        lambda_conv_ker=LAMBDA_CONV_KER,
        lambda_conv_act=LAMBDA_CONV_ACT,
        lambda_out_ker=LAMBDA_OUT_KER,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        batch_size=32,
        patience=PATIENCE,
        estimate_effects=True,
        n_bootstrap=1000,
        run_benchmarks=(TASK == "binary"),  # RLR baseline is for binary outcomes
    )

    InfluentialTextPipeline.print_summary(result)

    # --- Save results -------------------------------------------------------
    import torch, json
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(result.model.state_dict(), os.path.join(output_dir, "model.pt"))

    if result.interpretation is not None:
        rows = []
        for fi in result.interpretation.filters:
            phrases = [p["text"] for p in fi.top_phrases] if fi.top_phrases else []
            rows.append({
                "filter_id": fi.filter_idx,
                "kernel_size": fi.kernel_size,
                "output_weight": fi.output_weight,
                "treatment_effect": fi.effect_estimate,
                "ci_lower": fi.effect_ci[0] if fi.effect_ci else None,
                "ci_upper": fi.effect_ci[1] if fi.effect_ci else None,
                "activation_range": fi.activation_range,
                "is_active": fi.is_active,
                **{f"top_phrase_{i+1}": (phrases[i] if i < len(phrases) else "")
                   for i in range(5)},
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(output_dir, "filter_interpretations.csv"), index=False
        )

    summary = {
        "test_metrics": result.test_metrics,
        "cnn_r2_adj": result.cnn_r2_adj,
        "cnn_mse": result.cnn_mse,
    }
    if result.benchmark_results:
        for name, br in result.benchmark_results.items():
            summary[f"{name}_r2_adj"] = br.r_squared_adj
            summary[f"{name}_mse"]    = br.mse
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
