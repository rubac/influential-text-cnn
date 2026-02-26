# Influential Text CNN

A Python implementation of the CNN-based method for discovering influential text features, as described in:

> Ayers, Sanford, Roberts & Yang (2024). **"Discovering influential text using convolutional neural networks."** *Findings of the Association for Computational Linguistics: ACL 2024*, pp. 12002–12027. [[Paper]](https://aclanthology.org/2024.findings-acl.714/)

## What this does

Given a corpus of texts and a binary outcome (e.g., whether a social media post was censored, whether a complaint received a timely response), this method discovers **clusters of phrases that are predictive of the outcome**. Unlike topic models or bag-of-words approaches, the CNN can capture:

- Phrases of flexible length (not fixed n-grams)
- Grammatical and structural patterns (e.g., use of infinitive verbs, contractions)
- Contextual meaning via BERT embeddings

The discovered phrase clusters can serve as candidate **text treatments** for follow-up causal inference studies.

## Architecture

```
Text → BERT embeddings (N × U × D)
         ↓
   ┌─────────────┐   ┌─────────────┐
   │ Conv1D (K=5) │   │ Conv1D (K=7) │    ← parallel layers, sigmoid activation
   └──────┬──────┘   └──────┬──────┘
          ↓                  ↓
     Max pooling        Max pooling        ← per filter, across all phrases
          ↓                  ↓
          └────────┬─────────┘
             Concatenate
                 ↓
           Output layer → P(outcome = 1)
```

**Custom loss function:**
```
L = BCE + λ₁ · ‖W_conv‖²  +  λ₂ · max(R)  +  λ₃ · ‖W_out‖₁
         L2 on conv weights    filter diversity   L1 on output
```

where max(R) penalizes the maximum positive pairwise correlation between filter activations, encouraging each filter to learn a distinct pattern.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/rubac/influential-text-cnn.git
cd influential-text-cnn
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3):** The default `pip install torch` includes MPS (GPU) support. The pipeline auto-detects your GPU.

**NVIDIA GPU:** See [PyTorch install instructions](https://pytorch.org/get-started/locally/) for your CUDA version.

### 2. Prepare your data

You need a CSV with a text column and a binary (0/1) outcome column:

| text | outcome |
|---|---|
| The bank refused my application... | 0 |
| I received great service today... | 1 |

### 3. Run

**Option A — Edit the script:**

Open `run_analysis.py`, change the three lines at the top, and run:

```bash
python run_analysis.py
```

**Option B — Use the notebook:**

Open `notebooks/tutorial.ipynb` in Jupyter for a step-by-step walkthrough.

**Option C — Use in your own code:**

```python
from influential_text_cnn import InfluentialTextPipeline

pipeline = InfluentialTextPipeline(
    model_name="prajjwal1/bert-tiny",
    max_tokens=150,
)

result = pipeline.run(
    texts=texts,
    labels=labels,
    num_filters=8,
    kernel_sizes=[5, 7],
)

# Print discovered text treatments
InfluentialTextPipeline.print_summary(result)
```

### 4. Interpret results

The main output is `results/filter_interpretations.csv`. Each row is a learned filter (text pattern):

| Column | Meaning |
|---|---|
| `output_weight` | Direction and strength of association with outcome |
| `treatment_effect` | Estimated effect (under the paper's assumptions) |
| `ci_lower`, `ci_upper` | 95% bootstrap confidence interval |
| `is_active` | Whether the filter learned a meaningful pattern |
| `top_phrase_1`–`5` | Most strongly activating phrases |

**In R:**
```r
results <- read.csv("results/filter_interpretations.csv")
results <- results[order(-abs(results$output_weight)), ]
head(results)
```

## Reproducing the paper's applications

### Weibo censorship (Section 5.1)

```python
pipeline = InfluentialTextPipeline(
    model_name="hfl/chinese-roberta-wwm-ext",
    max_tokens=150,
)
result = pipeline.run(
    texts=weibo_texts, labels=censorship_labels,
    num_filters=8, kernel_sizes=[5, 7],
    lambda_conv_ker=0.001, lambda_conv_act=3.0,
    lambda_out_ker=0.0001, learning_rate=0.0001,
)
```

### CFPB complaints (Section 5.2)

```python
pipeline = InfluentialTextPipeline(
    model_name="prajjwal1/bert-tiny",
    max_tokens=250,
)
result = pipeline.run(
    texts=complaint_texts, labels=timely_response,
    num_filters=16, kernel_sizes=[5],
    lambda_conv_ker=0.0, lambda_conv_act=0.5,
    lambda_out_ker=0.001, learning_rate=0.001,
)
```

## Hyperparameter tuning

The paper tunes over a grid of model configurations using 5-fold CV. To enable this:

```python
result = pipeline.run(
    texts=texts, labels=labels,
    tune=True, n_folds=5,
)
```

This searches over filter counts, kernel sizes, regularization strengths, and learning rates. Model selection balances accuracy, filter diversity, and the number of useful filters (see Section 4.3 of the paper).

**Warning:** Tuning is compute-intensive. The paper reports 375 CPU-hours for 486 configurations on the Weibo data. Start with `tune=False` and fixed parameters.

## Project structure

```
influential-text-cnn/
├── influential_text_cnn/       # Python package
│   ├── __init__.py
│   ├── model.py                # InfluentialTextCNN + custom loss
│   ├── embedding.py            # BERT embedding wrapper
│   ├── training.py             # Training loop + early stopping
│   ├── tuning.py               # Hyperparameter grid search with CV
│   ├── interpretation.py       # Filter phrase extraction + treatment effects
│   ├── benchmarks.py           # Regularized logistic regression baseline
│   ├── pipeline.py             # End-to-end pipeline
│   └── visualization.py        # Plotting utilities
├── notebooks/
│   └── tutorial.ipynb          # Step-by-step tutorial
├── run_analysis.py             # Simple script entry point
├── requirements.txt
├── LICENSE
└── README.md
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `bert_model` | `prajjwal1/bert-tiny` | HuggingFace BERT model for embeddings |
| `max_tokens` | 150 | Max tokens per text (increase for longer docs) |
| `num_filters` | 8 | Filters per conv layer (4, 8, or 16) |
| `kernel_sizes` | `[5, 7]` | Phrase lengths in tokens |
| `lambda_conv_ker` | 0.001 | L2 penalty on conv weights |
| `lambda_conv_act` | 3.0 | Filter diversity penalty (higher → more diverse) |
| `lambda_out_ker` | 0.0001 | L1 penalty on output weights |
| `learning_rate` | 0.001 | Adam learning rate |
| `epochs` | 100 | Max training epochs |
| `patience` | 15 | Early stopping patience |

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ayers-etal-2024-discovering,
    title     = "Discovering Influential Text Using Convolutional Neural Networks",
    author    = "Ayers, Megan and Sanford, Luke and Roberts, Margaret E. and Yang, Eddie",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    year      = "2024",
    pages     = "12002--12027",
    url       = "https://aclanthology.org/2024.findings-acl.713",
}
```

## License

MIT
