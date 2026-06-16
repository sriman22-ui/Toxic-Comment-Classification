# Toxic Comment Multi-Label Classification

Detecting and classifying toxic comments from Wikipedia across six categories using a classic NLP pipeline: **TF-IDF + engineered features → Logistic Regression** (multi-label, one-vs-rest).

> IE0005 Mini Project — Team DSAI EL17 (Sriman, Aravinth, Lavantika, Louise)

## Overview

Online comments can be toxic in several distinct ways at once — a single comment might be `toxic`, `obscene`, *and* an `insult`. This project builds a machine learning pipeline that flags six toxicity categories simultaneously:

`toxic` · `severe_toxic` · `obscene` · `threat` · `insult` · `identity_hate`

It is a **multi-label** problem (labels can co-occur), not a single-label one. The model is trained on ~127K cleaned Wikipedia comments and reaches a mean ROC-AUC of **0.9735** on the validation set.

## Dataset

[Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data). Three files are required:

| File | Description |
|---|---|
| `train.csv` | 159,571 labelled Wikipedia comments (training) |
| `test.csv` | 153,164 unlabelled comments (predictions) |
| `test_labels.csv` | Ground-truth labels for the test set (rows marked `-1` are excluded from evaluation) |

These files are **not** included in this repo (see [Setup](#setup)). Download them from Kaggle and place them in the project root.

## Pipeline

1. **EDA** — class imbalance (~90% non-toxic), label distribution, and a label correlation heatmap that justifies the multi-label approach.
2. **Data cleaning** — lowercasing, HTML stripping, contraction expansion, removal of numbers/punctuation, and dropping empty comments. Negation words (`no`, `not`, `nor`, `never`) are deliberately preserved.
3. **Feature engineering** — five hand-crafted signals alongside the text: `char_count`, `word_count`, `exclamation_count`, `question_count`, `uppercase_ratio`.
4. **Vectorisation** — TF-IDF (`max_features=30000`, `ngram_range=(1,2)`, custom stopwords, `sublinear_tf`), combined with `StandardScaler`-normalised engineered features via sparse `hstack`.
5. **Modelling**
   - *Baseline:* Multinomial Naive Bayes (TF-IDF only) — mean ROC-AUC **0.9567**
   - *Main model:* Logistic Regression with `class_weight='balanced'` (one-vs-rest) — mean ROC-AUC **0.9735**
6. **Analysis** — per-label threshold tuning, model interpretability (inspecting learned weights), and a bias/fairness evaluation of identity-term false positive rates.
7. **Optimisation** — stratified split, custom RandomOverSampler, and hyperparameter tuning of `C`.

## Results

| Model | Features | Mean ROC-AUC |
|---|---|---|
| Naive Bayes (baseline) | TF-IDF only | 0.9567 |
| Logistic Regression (main) | TF-IDF + 5 engineered | **0.9735** |

Logistic Regression outperforms the baseline on every label, with the largest gains on the rarest categories (`threat`, `obscene`). The pipeline deliberately favours **recall over precision** — for content moderation, missing a genuine threat is more costly than over-flagging a borderline comment.

### Bias finding

The fairness evaluation surfaced systematic bias: non-toxic comments mentioning certain identity terms are wrongly flagged far more often than the overall baseline (FPR 0.1227). For example, `gay` reaches an FPR of 0.5696 (~4.6× baseline), and a notable gender asymmetry exists between `woman` (0.3349) and `man` (0.1276). The root cause is spurious correlation in the training data — the model learned to associate identity terms themselves with toxicity.

## Setup

```bash
# 1. Clone
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset from Kaggle and place these in the project root:
#    train.csv  test.csv  test_labels.csv
```

Then open the notebook:

```bash
jupyter notebook Project_Code_Final.ipynb
```

Run the cells top to bottom. Cleaning is cached to `train_cleaned.csv` / `test_cleaned.csv` so it doesn't need to be re-run.

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn, scipy, matplotlib, seaborn

A `requirements.txt` is included.

## Limitations & future work

- **Bag-of-words:** word order is ignored, so "I don't hate you" and "I hate you" look similar. A transformer (BERT/DistilBERT) would address this.
- **Bias:** threshold tuning mitigates but doesn't eliminate identity-term associations; adversarial debiasing would tackle the root cause.
- **Rare labels:** `threat` and `identity_hate` have few examples; SMOTE or richer data collection could help.

## Team contributions

- **Sriman** — dataset selection, EDA, visualisation, cleaning, feature engineering, TF-IDF
- **Aravinth** — Naive Bayes baseline, Logistic Regression, threshold pipeline, bias evaluation
- **Lavantika** — precision/recall/F1/ROC-AUC evaluation, threshold analysis, model comparison
- **Louise** — optimisation pipeline (stratified split, oversampling, hyperparameter tuning)
