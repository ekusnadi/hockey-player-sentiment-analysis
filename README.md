# Hockey Player Sentiment Analysis

## Author & Submission Info
- **Author:** Ethan Kusnadi  
- **Date of Submission:**  May 22, 2026
- **Oral Presentation Recording:** https://drive.google.com/file/d/1iB8le56cZkfxx2Xr69V9zvB3MYQ9Ok0u/view?usp=sharing

## Overview
A Python NLP pipeline that scrapes Reddit hockey game threads, identifies player mentions, classifies sentiment, and generates per-player sentiment scores as a complement to the box score.

## Project Structure
- **data/**
  - `raw/` — scraped Reddit comments CSV files
  - `training/` — labeled training data and redacted comments
  - `output/` — confusion matrices, mislabeled comments, and sentiment visualizations
- **src/**
  - `scrape_all_comments.py` — scrapes Reddit game thread comments
  - `redact_names.py` — redacts player names using fuzzy matching
  - `sharks_aliases.py` — player name and alias definitions
  - `roberta_finetuning.py` — fine-tunes RoBERTa on labeled comments
  - `model_testing.py` — evaluates VADER, TextBlob, and RoBERTa baselines
- `Sentiment_Analysis_of_Hockey_Players.pdf` — written project report
- `hockey_sentiment_pipeline.ipynb` — end-to-end sentiment analysis pipeline

## Requirements
```bash
pip install transformers torch scikit-learn pandas nltk textblob scipy matplotlib adjustText nhlpy rapidfuzz
```

## Usage
> **Note:** The fine-tuned model is not included in this repository due to file size. Run step 1 to generate it locally before running the pipeline.

After cloning the repository, move into the project folder:
```bash
git clone https://github.com/ekusnadi/hockey-player-sentiment-analysis.git
cd hockey-player-sentiment-analysis
```

1. Build fine-tuned RoBERTa model: `python src/roberta_finetuning.py`
2. Run the pipeline by opening and running: `hockey_sentiment_pipeline.ipynb`

## Results
Evaluation outputs including confusion matrices and mislabeled comments are saved in `data/output/`.

| Model | Accuracy | Macro F1 |
|---|---|---|
| VADER (base) | 0.44 | 0.44 |
| VADER (augmented) | 0.46 | 0.46 |
| TextBlob | 0.43 | 0.43 |
| RoBERTa (zero-shot) | 0.63 | 0.63 |
| RoBERTa (fine-tuned) | 0.76 | 0.74 |
