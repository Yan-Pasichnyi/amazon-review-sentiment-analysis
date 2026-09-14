# Amazon Review Sentiment Analysis

Binary sentiment classification on ~520K Amazon Fine Food Reviews, covering the full NLP pipeline: 
exploratory analysis, text preprocessing, model development, and structured error analysis.

## Project overview

Given a review's text, predict whether it expresses positive or negative sentiment. Built as an 
end-to-end portfolio project to demonstrate NLP fundamentals — text cleaning, feature engineering, 
rigorous model selection, and honest evaluation of a model's limitations.

**Dataset**: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) 
(Kaggle) — ~568K reviews with text, star ratings (1–5), and metadata.

## Results

**Final model**: LinearSVC on TF-IDF with bigrams (`max_features=10000`, `ngram_range=(1, 2)`, `C=1`), 
selected via `GridSearchCV` with 3-fold cross-validation.

| Metric | Negative class | Positive class |
|---|---|---|
| Precision | 0.86 | 0.96 |
| Recall | 0.76 | 0.98 |
| F1 | 0.81 | 0.97 |

**Macro F1: 0.89** (cross-validated F1 on train: 0.884 — closely matching the held-out test result, 
confirming the model generalizes rather than having overfit to a particular data split)

## Pipeline & key decisions

**1. EDA** — identified severe class imbalance (64% of reviews are 5-star), HTML artifacts 
(`<br />`, `<a href>`) from web scraping, and 281 duplicate reviews.

![Score distribution](images/histogram_of_score_column.png)

**2. Target definition** — converted to binary sentiment (Score 4-5 → positive, 1-2 → negative), 
dropping neutral 3-star reviews rather than treating them as a third class, since mixed sentiment 
in neutral reviews would blur the decision boundary without adding practical value.

**3. Text preprocessing** (spaCy, `en_core_web_sm`) — two deliberate decisions that shaped the 
whole pipeline:
- **Punctuation was preserved before tokenization.** Stripping it first breaks contractions 
  ("didn't" → "didn", "t"), losing sentiment-critical negation. spaCy's tokenizer handles 
  contractions correctly on its own; punctuation is removed afterward via `token.is_alpha`.
- **Negation words were excluded from the stop word list.** spaCy treats "not", "no", and "never" 
  as stop words by default — removing them silently flips the meaning of phrases like "did not 
  have any results" into a neutral bag of words.

**4. Modeling** — established a TF-IDF + Logistic Regression baseline (macro F1 0.85), then 
compared class weighting, bigram features, vocabulary size, and algorithm choice. Model selection 
was performed using `Pipeline` + `GridSearchCV` with 3-fold cross-validation, rather than by 
repeatedly evaluating each configuration against the held-out test set — the latter risks 
selection bias, since the "winning" configuration could partly reflect quirks of that specific 
test split rather than a genuine improvement. The test set was touched exactly once, for a 
single final evaluation of the selected model. (An earlier, manual comparison had already 
converged on the same configuration — the more rigorous process confirmed rather than changed 
the result.)

**5. Error analysis** — surfaced two distinct, non-overlapping failure modes:
- **Mixed-sentiment reviews**: text that opens positive and pivots negative (or vice versa). 
  TF-IDF's bag-of-words representation has no concept of word order, so it can't weigh a later 
  sentiment shift the way a human reader naturally would.
- **Label noise**: some reviews' text and star rating are simply inconsistent (e.g. describing 
  a real problem but still rating 5 stars) — a ceiling no text-only model can fully overcome.

## Tech stack

Python, pandas, NumPy, scikit-learn, spaCy, matplotlib, seaborn, joblib

## Project structure
```
amazon-review-sentiment-analysis/
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_text_preprocessing.ipynb
│ ├── 03_tfidf_baseline.ipynb
│ ├── 04_model_improvements.ipynb
│ └── 05_error_analysis.ipynb
├── images/
│ └── histogram_of_score_column.png
├── models/
│ └── sentiment_model.pkl # saved final pipeline (TF-IDF + LinearSVC)
├── data/raw/ # not tracked — see setup below
├── requirements.txt
└── README.md
```


## Setup

```powershell
git clone https://github.com/Yan-Pasichnyi/amazon-review-sentiment-analysis.git
cd amazon-review-sentiment-analysis
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Download the [dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) from Kaggle 
and place `Reviews.csv` in `data/raw/`.

## Possible next steps

- Transformer-based models (e.g. fine-tuned BERT via HuggingFace) to capture word order and 
  context — would likely address the mixed-sentiment failure mode identified in error analysis
- Multi-class classification (predicting the actual 1-5 star rating, not just binary sentiment)