# Representation Learning for Temporal Customer Behavior

## Motivation

This project began from a simple question:

> How much does optimization objective performance matter when evaluating a self-supervised representation learning model, and if the learned representations are meaningful, can they capture temporal customer behavior from transaction sequences?

The deeper idea is not just whether a model predicts the next transaction well, but whether the model learns a useful representation of each customer or account behavior over time. In other words, do the learned embeddings reflect real behavioral structure, even when the downstream prediction objective is imperfect or only loosely aligned with the underlying concept we care about?

This is especially relevant in temporal or sequential data settings, where the objective may be to predict the next transaction, yet the real value lies in the representation itself: clusters, similarities, anomalies, and latent customer profiles.

---

## Research Framing

The project tests a practical hypothesis:

- A self-supervised encoder trained on account-level transaction sequences can learn meaningful representations.
- These representations may capture behavioral patterns beyond the immediate forecasting target.
- The quality of the representation should be evaluated not only by objective metrics, but also by whether the embeddings show structure that is interpretable and useful downstream.

In this repo, the optimization task is framed around next-transaction forecasting, while the representation is the central object of interest. The model is trained to predict a future transaction amount and temporal gap, but the learned account embedding is treated as the core artifact for understanding customer behavior.

---

## What this repository does

This project explores whether sequential transaction histories can be compressed into account-level embeddings using a transformer-based encoder. The model is trained on sliding windows of customer transactions and learns to generate a latent representation for each account based on their temporal spending behavior.

The workflow includes:

- loading and preprocessing synthetic banking transaction data
- creating per-account time-ordered sequences
- generating sliding windows for supervised next-event prediction
- building a transformer encoder architecture
- extracting account embeddings from the model
- clustering customers using the learned representations
- evaluating whether embeddings reveal behaviorally meaningful groups or patterns

---

## Problem setting

The data consists of transaction records for accounts, with fields such as:

- AccountID
- CustomerAge
- CustomerOccupation
- TransactionAmount
- TransactionType
- MerchantID
- Channel
- LoginAttempts
- DateDifference
- TransactionDate

The goal is not to build a production-grade fraud detector or financial predictor. Instead, the focus is on representation learning and hypothesis testing in a sequential transaction setting.

This repository is intentionally exploratory: it asks whether a model trained to predict the next transaction can also produce latent embeddings that correspond to meaningful customer segments.

---

## Hypothesis

If an encoder learns from a sequence of transactions, then the resulting account embeddings should contain information about:

- spending patterns
- transaction cadence
- channel usage
- customer behavior variation across accounts
- latent customer segments or clusters

The real question is whether these embeddings retain structure that remains useful even if objective performance is modest.

This project evaluates the idea that optimization loss is only one lens. A model may have imperfect predictive accuracy yet still learn useful representations.

---

## Methodology

### 1. Data preparation

- Sort transactions by account and time
- Drop non-essential identifiers
- Extract temporal features such as transaction day and month
- Compute inter-transaction time gaps using DateDifference
- Split each account history into train/test windows

### 2. Feature encoding

- Label encode categorical fields
- Standardize numerical features
- Build a sliding-window dataset where each sample is a sequence of transactions used to predict the next transaction target

### 3. Self-supervised sequence model

The model uses a transformer-style encoder with:

- categorical embeddings for transaction attributes
- positional encoding for transaction order
- multi-head attention blocks
- pooling over the sequence to produce a single account representation
- prediction heads for regression targets

### 4. Representation extraction

The encoder produces an account embedding from the windowed transaction sequence. These embeddings are then analyzed for:

- cluster structure
- similarity patterns across customers
- representation stability
- whether latent behavior groups emerge without explicit labels

---

## Repository structure

```text
Representation Learning/
├── Notebook/
│   └── Representation_Learning.ipynb
├── transformer_ach/
│   └── Encoder_Achitecture.py
├── data/
│   └── bank_transactions_data_2_augmented_clean_2.csv
├── Artifacts/
│   ├── Representational_Learning.pt
│   ├── feature_arangement.txt
│   ├── label_encoders.pkl
│   └── standard_scalers.pkl
├── README.md
└── .git/
```

---

## Dependencies

This project uses:

- Python 3.10+
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib

Example installation:

```bash
pip install pandas numpy torch matplotlib seaborn scikit-learn joblib
```

---

## How to run

Open and run the notebook in:

```text
Notebook/Representation_Learning.ipynb
```

The notebook contains the full pipeline, including:

- preprocessing
- sliding-window construction
- model definition
- training loop
- representation extraction
- clustering and evaluation

---

## Evaluation lens

The project is designed to question a common assumption in representation learning:

> A model that performs poorly on the optimization objective is not necessarily learning poor representations.

In this setting, the objective is next-transaction prediction. But the downstream question is different:

- Are the learned account embeddings useful?
- Do they separate behavioral groups?
- Do they capture meaningful differences between customers?
- Are the representations structurally coherent even when prediction error remains relatively high?

This makes the repo an investigation into representation quality as a research question rather than a pure supervised forecasting problem.

---

## Key takeaway

This repository explores a central question in self-supervised learning:

What matters more for evaluating a representation learning model: the optimization objective, or the structure and usefulness of the learned embeddings themselves?

In temporal customer behavior data, the answer is likely more nuanced than a single metric can capture.

---

## Notes

- The dataset is synthetic.
- The project is exploratory and research-oriented.
- The main emphasis is on testing a representation-learning hypothesis, not on claiming production-grade prediction performance.
- This repo is best understood as a prototype for studying whether temporal self-supervised learning can yield useful behavioral embeddings.

---

## Status

This project is a research prototype and experimental notebook-driven implementation. It is intended to support hypothesis testing, not an industrial pipeline.

---

## Citation / context

This repo reflects a personal experiment in representational learning for customer behavior and temporal data. The core motivation is to investigate how optimization metrics and learned representations relate when the real object of interest is latent behavioral structure.

---

## A simple one-line summary

Self-supervised temporal representation learning for customer transaction behavior, with the goal of understanding whether next-event prediction can produce meaningful account embeddings.
