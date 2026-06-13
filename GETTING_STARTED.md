# Getting started

A step-by-step guide for running the Influential Text CNN on your own data.
It assumes you can use a terminal and write a little Python, but it does **not**
assume any prior experience with PyTorch, BERT, or neural networks.

If you'd rather learn by reading runnable code, open
[`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) instead — it covers the
same ground interactively.

---

## What problem does this solve?

You have a collection of **texts** (survey open-ends, social-media posts, news
articles, complaints, …) and, for each text, a **outcome** you care about —
something binary (was the post removed? did the complaint get a timely reply?)
or numeric (a sentiment score, a response time).

You want to know: **which ways of phrasing things are associated with the
outcome?** Not just single words, but multi-word patterns — "demand a refund",
the use of polite hedges, a particular grammatical construction.

This tool learns a small set of **phrase detectors** ("filters") that are
predictive of the outcome, and shows you the phrases from your own corpus that
each one picks up. Those phrases are *candidate text treatments* you can then
test in a follow-up experiment.

It is **not** a topic model and **not** a bag-of-words classifier — it reads
short spans of text in context (via BERT) and can capture patterns those
methods miss.

---

## 1. Install

You need Python 3.9 or newer. From the project folder:

```bash
pip install -r requirements.txt
```

This installs PyTorch, Hugging Face Transformers, scikit-learn, pandas, and
matplotlib. The first run will also download a small BERT model (~20 MB for
`bert-tiny`).

- **Apple Silicon (M1/M2/M3):** the default install includes GPU (MPS) support;
  the pipeline auto-detects it.
- **NVIDIA GPU:** follow the [PyTorch install guide](https://pytorch.org/get-started/locally/)
  for your CUDA version if you want GPU acceleration.
- **No GPU:** that's fine. `bert-tiny` runs comfortably on a laptop CPU.

To check the install worked:

```bash
python -c "from influential_text_cnn import InfluentialTextPipeline; print('OK')"
```

---

## 2. Prepare your data

Put your data in a CSV (or TSV / XLSX) with at least two columns: one for the
text, one for the outcome.

| text | outcome |
|---|---|
| The bank refused my application without explanation... | 0 |
| I received great service today, thank you! | 1 |

Rules of thumb:

- **Binary outcome:** code it as `0` / `1`.
- **Continuous outcome:** any number is fine (you'll set `TASK = "continuous"`).
- **Size:** the method works on a few hundred texts but is happier with a few
  thousand. Class balance matters less than having enough examples of each
  outcome.
- **Cleaning:** light cleaning only. Don't strip punctuation or stopwords — the
  model uses them. Just remove rows with empty text (the script does this for you).

---

## 3. Run it

### The no-code path

1. Open `run_analysis.py` in any editor.
2. Edit **Step 1** to point at your file and name your two columns:
   ```python
   DATA_FILE      = "my_complaints.csv"
   TEXT_COLUMN    = "body"
   OUTCOME_COLUMN = "was_escalated"
   ```
3. (Optional) In **Step 3**, set `TASK = "continuous"` if your outcome is numeric.
4. Run it:
   ```bash
   python run_analysis.py
   ```

It will embed your texts, train the model, print a summary, and save results to
the `results/` folder.

### Tip: start small

The first time, run on a **subset** (say 500 rows) with the default
`prajjwal1/bert-tiny` model just to confirm everything works end-to-end. Then
switch to `bert-base-uncased` (or a language-specific model) and the full data
for your real results.

---

## 4. Read the results

Three files land in `results/`:

- **`filter_interpretations.csv`** — the main output. One row per learned phrase
  pattern.
- **`metrics.json`** — how well the model fit, plus the n-gram baseline for comparison.
- **`model.pt`** — the trained model weights (so you can reload without retraining).

Open `filter_interpretations.csv` in R, Excel, or pandas. For each **active**
filter (`is_active == True`):

1. **Read `top_phrase_1` … `top_phrase_5`.** These are real snippets from your
   corpus. They tell you *what pattern the filter detects* — give it a short label
   in your own words ("polite request", "threat to escalate").
2. **Check `output_weight`.** Positive → the pattern goes with `outcome = 1` (or
   higher values); negative → with `outcome = 0`. Use the magnitude only to rank
   filters, not as an effect size.
3. **Check `treatment_effect` and the CI** (`ci_lower`, `ci_upper`). This is an
   OLS estimate on held-out data. A CI that excludes 0 means the association is
   distinguishable from zero in your sample.

> **Important — discovery, not proof.** The `treatment_effect` is a *causal*
> quantity only under the strong assumptions in Section 3 of the paper. Treat the
> output as a list of **hypotheses**: phrasings worth testing in a controlled
> experiment, not established causal effects. Filters can correlate with each
> other and with confounders.

---

## 5. Common questions

**It's slow / runs out of memory.**
Lower `MAX_LENGTH` (fewer tokens per text), keep `BERT_MODEL = "prajjwal1/bert-tiny"`,
or run on a subset first. Memory scales with `max_tokens` × embedding size × number
of texts.

**My texts are long (articles, transcripts).**
Set `MAX_LENGTH` to roughly your longest document's token count (e.g. `14000`).
Texts longer than 510 tokens are automatically split into overlapping chunks,
embedded separately, and stitched back together — no text is lost.

**All my filters are inactive / they all look the same.**
See the "Regularization guidance" in the tutorial notebook (Section 8). Briefly:
lower `LAMBDA_CONV_ACT`/`LAMBDA_CONV_KER` if everything is inactive; raise
`LAMBDA_CONV_ACT` if filters are redundant.

**Should I use hyperparameter tuning (`tune=True`)?**
Not at first. It runs a k-fold cross-validated grid search and is very slow (the
paper reports hundreds of CPU-hours). Get a sensible result with the defaults,
then tune only if you need to squeeze out performance.

**Which BERT model should I use?**

| Language | Fast | Better quality |
|---|---|---|
| English | `prajjwal1/bert-tiny` | `bert-base-uncased` |
| Chinese | `bert-base-chinese` | `hfl/chinese-roberta-wwm-ext` |
| Other | `bert-base-multilingual-cased` | a language-specific model from Hugging Face |

---

## Where to go next

- [`notebooks/tutorial.ipynb`](notebooks/tutorial.ipynb) — the same workflow,
  interactively, plus plots and a continuous-outcome example.
- [`README.md`](README.md) — architecture, the loss function, and the parameter reference.
- The [paper](https://aclanthology.org/2024.findings-acl.714/) — the method and its assumptions in full.
