# Male Infertility Disclosure on Reddit

> **Study under preparation:** *Male Infertility Disclosure on Reddit: A Study of Emotional Expression, Information Exchange, and Peer Support*

This repository contains the data collection and analysis code for a computational study examining how individuals discuss male infertility on Reddit. The study investigates emotional expression, information-seeking behaviour, peer support dynamics, and topical patterns within the r/maleinfertility community.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Scripts](#scripts)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

---

## Overview

Reddit communities provide a unique window into patient-reported experiences that are often absent from clinical literature. This study focuses on the r/maleinfertility subreddit, applying natural language processing (NLP) methods to characterise:

- **Sentiment** – whether posts and comments express positive, neutral, or negative affect.
- **Emotion** – fine-grained emotional classification (anger, disgust, fear, joy, neutral, sadness, surprise).
- **Topics** – latent themes discussed by community members.
- **Specific medical concepts** – mentions of fertility tests, medications, and donor conception.
- **User engagement** – longitudinal activity patterns of community members.

---

## Repository Structure

```
.
├── posts.py               # Convert raw posts JSONL → CSV
├── comments.py            # Convert raw comments JSONL → CSV
├── process_paragraph.py   # Text pre-processing pipeline
├── helpers.py             # Shared utilities (backup, abbreviations)
├── analyse_sentiment.py   # VADER sentiment analysis
├── emotion.py             # Transformer-based emotion classification
├── topic_model_save.py    # BERTopic topic modelling
├── generate_wordcloud.py  # Word cloud generation
├── shift_analysis.py      # Word-shift graph (posts vs. comments)
├── extract_links.py       # URL extraction and domain counting
├── analyse_drug.py        # Drug/medication mention analysis
├── analyse_donor.py       # Donor conception mention analysis
├── analyse_test.py        # Fertility test mention analysis
└── track_users.py         # Longitudinal user activity analysis
```

---

## Scripts

### Data Extraction

| Script | Description |
|--------|-------------|
| `posts.py` | Reads Reddit posts from JSONL (Pushshift format), filters removed/deleted entries, and writes a structured CSV with fields: ID, Author, Title, Body, Date, Time, Subreddit, URL, Upvotes, Score, Awards, Comments, Crosspost. |
| `comments.py` | Reads Reddit comments from JSONL, filters removed/deleted comments and AutoModerator entries, and writes a CSV with fields: ID, Author, Body, Date, Time, Subreddit, URL, Parent_ID, Upvotes, Awards. |

### Text Pre-processing

| Script | Description |
|--------|-------------|
| `process_paragraph.py` | Shared pre-processing pipeline: demojisation, emoticon translation, URL/email/username/hashtag removal, abbreviation expansion (100+ entries), spell-checking (pyspellchecker), WordNet lemmatisation, and stop-word removal. |
| `helpers.py` | Utility functions: `backup()` (timestamped file backup before overwriting) and `get_abbreviations()` (abbreviation dictionary). |

### NLP Analysis

| Script | Description |
|--------|-------------|
| `analyse_sentiment.py` | Applies VADER (`nltk`) to classify each post/comment as Positive (compound ≥ 0.05), Negative (compound ≤ −0.05), or Neutral, and appends the compound score. |
| `emotion.py` | Runs the `j-hartmann/emotion-english-distilroberta-base` transformer model to predict one of seven emotions per text (anger, disgust, fear, joy, neutral, sadness, surprise) and appends per-class probability scores. |
| `topic_model_save.py` | Builds a BERTopic model using `all-MiniLM-L6-v2` sentence embeddings, UMAP dimensionality reduction, HDBSCAN clustering, and KeyBERT-inspired topic representation. Saves the model and a human-readable topic summary. |

### Visualisation

| Script | Description |
|--------|-------------|
| `generate_wordcloud.py` | Generates word cloud images from pre-processed text. Supports standard frequency-based mode and a `--tfidf` flag for TF-IDF-weighted clouds. |
| `shift_analysis.py` | Produces a proportion word-shift graph comparing word frequencies between posts and comments using the `shifterator` library. |

### Thematic Analyses

| Script | Description |
|--------|-------------|
| `analyse_drug.py` | Matches drug/medication keywords (loaded from `Keywords.xlsx`) against post/comment bodies and outputs sentiment and emotion breakdowns per drug. |
| `analyse_donor.py` | Identifies donor conception mentions (donor sperm, sperm bank, IUI/IVF with donor) using three keyword lists, and produces per-list sentiment and emotion summaries. |
| `analyse_test.py` | Matches fertility test keywords (loaded from `MaleInfertility_Test_MK290925.xlsx`) and outputs sentiment and emotion breakdowns per test. |

### User Analysis

| Script | Description |
|--------|-------------|
| `track_users.py` | Tracks longitudinal activity of community members, computing yearly post and comment counts per user, and reporting the distribution of active months. Users must be active for a configurable minimum number of months (`MIN_ACTIVE_MONTHS`, default: 10) to be included. |

### Supplementary

| Script | Description |
|--------|-------------|
| `extract_links.py` | Extracts URLs from post/comment bodies, resolves redirects, and produces a domain frequency count CSV. |

---

## Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies include:

- `nltk` – VADER sentiment analysis, tokenisation, stop words
- `transformers`, `torch` – emotion classification model
- `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan` – topic modelling
- `wordcloud`, `matplotlib` – word cloud generation
- `shifterator` – word-shift analysis
- `pyspellchecker` – spell correction
- `pandas`, `tqdm`, `requests`, `emoji`, `openpyxl` – data handling and utilities

---

## Usage

### 1. Extract posts and comments

```bash
python posts.py data/r_maleinfertility_posts.jsonl
python comments.py
```

### 2. Run sentiment analysis

```bash
python analyse_sentiment.py extracted/r_maleinfertility_posts.csv extracted/r_maleinfertility_comments.csv
```

### 3. Run emotion classification

```bash
python emotion.py extracted/r_maleinfertility_posts_sentiment.csv extracted/r_maleinfertility_comments_sentiment.csv
```

### 4. Topic modelling

```bash
python topic_model_save.py extracted/r_maleinfertility_posts_sentiment_emotion.csv
```

### 5. Visualisations

```bash
# Word cloud (frequency-based)
python generate_wordcloud.py extracted/r_maleinfertility_posts.csv

# Word cloud (TF-IDF weighted)
python generate_wordcloud.py --tfidf extracted/r_maleinfertility_posts.csv

# Word-shift graph
python shift_analysis.py
```

### 6. Thematic analyses

```bash
python analyse_drug.py extracted/r_maleinfertility_posts_sentiment_emotion.csv extracted/r_maleinfertility_comments_sentiment_emotion.csv
python analyse_donor.py extracted/r_maleinfertility_posts_sentiment_emotion.csv extracted/r_maleinfertility_comments_sentiment_emotion.csv
python analyse_test.py extracted/r_maleinfertility_posts_sentiment_emotion.csv extracted/r_maleinfertility_comments_sentiment_emotion.csv
```

### 7. User activity analysis

```bash
python track_users.py
# or with explicit paths:
python track_users.py extracted/r_maleinfertility_comments_sentiment_emotion.csv extracted/r_maleinfertility_posts_sentiment_emotion.csv
```

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file included in this repository.
