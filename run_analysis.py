"""
=============================================================================
ConTextNet — Run analysis on your own data
=============================================================================

Usage:
    1. Edit the CONFIGURATION section below (Steps 1–3)
    2. Run:  python run_analysis.py

See GETTING_STARTED.md for detailed instructions.
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

# Make the package importable from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from influential_text_cnn import InfluentialTextPipeline


# =============================================================================
# STEP 1: YOUR DATA
# =============================================================================

DATA_FILE       = "your_data.csv"     # <-- Path to your CSV / TSV / XLSX
TEXT_COLUMN     = "text"              # <-- Column with text
OUTCOME_COLUMN  = "outcome"           # <-- Column with binary outcome (0/1)

# =============================================================================
# STEP 2: BERT MODEL
# =============================================================================
# English:  "prajjwal1/bert-tiny" (fast) | "distilbert-base-uncased" | "bert-base-uncased"
# Chinese:  "bert-base-chinese" | "hfl/chinese-roberta-wwm-ext"
# Multi:    "bert-base-multilingual-cased"

BERT_MODEL = "prajjwal1/bert-tiny"

# =============================================================================
# STEP 3: MODEL PARAMETERS  (defaults match the paper)
# =============================================================================

MAX_LENGTH      = 150          # Max tokens per text
KERNEL_SIZES    = [5, 7]       # Phrase lengths (in tokens)
NUM_FILTERS     = 8            # Filters per conv layer
EPOCHS          = 100          # Max training epochs
PATIENCE        = 15           # Early-stopping patience
LEARNING_RATE   = 0.001
LAMBDA_CONV_KER = 0.001        # L2 on conv weights
LAMBDA_CONV_ACT = 3.0          # Filter diversity penalty
LAMBDA_OUT_KER  = 0.0001       # L1 on output weights

# =============================================================================
# (No changes needed below this line)
# =============================================================================

def main():
    import pandas as pd

    # --- Load ---------------------------------------------------------------
    logger.info(f"Loading {DATA_FILE}...")
    if DATA_FILE.endswith(".tsv"):
        df = pd.read_csv(DATA_FILE, sep="\t")
    elif DATA_FILE.endswith((".xls", ".xlsx")):
        df = pd.read_excel(DATA_FILE)
    else:
        df = pd.read_csv(DATA_FILE)

    for col, label in [(TEXT_COLUMN, "text"), (OUTCOME_COLUMN, "outcome")]:
        if col not in df.columns:
            sys.exit(
                f"ERROR: Column '{col}' not found.\n"
                f"Available columns: {list(df.columns)}"
            )

    texts  = df[TEXT_COLUMN].astype(str).tolist()
    labels = df[OUTCOME_COLUMN].values.astype(float)

    unique = set(np.unique(labels))
    if not unique.issubset({0.0, 1.0}):
        sys.exit(f"ERROR: Outcome must be 0/1. Found: {unique}")

    # Drop empties
    valid = [len(t.strip()) > 0 for t in texts]
    if not all(valid):
        n_drop = sum(not v for v in valid)
        logger.info(f"  Dropping {n_drop} empty texts")
        texts  = [t for t, v in zip(texts, valid) if v]
        labels = labels[np.array(valid)]

    logger.info(f"  {len(texts)} texts | {labels.mean():.1%} positive")

    # --- Pipeline -----------------------------------------------------------
    pipeline = InfluentialTextPipeline(
        model_name=BERT_MODEL,
        max_tokens=MAX_LENGTH,
        test_fraction=0.2,
    )

    result = pipeline.run(
        texts=texts,
        labels=labels,
        tune=False,
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
        run_benchmarks=True,
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
