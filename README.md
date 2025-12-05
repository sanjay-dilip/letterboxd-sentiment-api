# 🎬 Letterboxd Movie Sentiment API

This project builds a sentiment system for Letterboxd reviews. The goal is to handle messy, sarcastic, emotional reviews and still predict whether the writer felt positive or negative about a movie. The project covers data cleaning, handling weak labels, two models, a small human-evaluation set, and a FastAPI backend.

##  📌 Overview

Letterboxd reviews are not straightforward. People joke, use slang, flip meanings, and sometimes the star rating doesn’t match the text. The whole point of this project is to see how far we can get with:

- weak labels from star ratings
- a simple baseline model
- a transformer fine-tuned on this dataset
- a small set of human-labelled samples
- an API that serves the predictions

The pipeline puts everything together from raw data to an actual service you can hit with a request.

## 🗂️ Data

The dataset is a scraped collection of Letterboxd reviews with fields such as:

- movie name
- release year
- review text
- reviewer name
- broken Unicode star ratings
- review date
- like and comment counts

After cleaning, the processed dataset includes:

- numeric star ratings
- clean review text
- loose 3-class sentiment
- strict binary sentiment
- ~3500 total reviews
- ~514 strict weak-labeled reviews
- 49 manually labeled samples for evaluation

Processed files live in `data/processed/`.

## 🔧 Data Pipeline

The pipeline handles the full prep flow:

- load raw reviews
- fix corrupted star rating characters
- convert stars to numbers
- build strict binary labels
- clean the review text
- parse dates
- remove unusable rows
- save final data into data/processed

This keeps the rest of the project neat and reproducible.

## 📒 Notebooks

### 1. EDA

Looks at:
- rating spread
- strict label balance
- review length patterns
- sample reviews
- checks for any strange characters

### 2. Baseline Model Training

Covers:
- loading strict labels
- basic balancing
- TF-IDF vectorization
- Logistic Regression training
- simple validation

The baseline is used as a reference point for the transformer.

## 🤖 Models

### Baseline: TF-IDF + Logistic Regression

This model:
- uses strict binary labels
- vectorizes text with TF-IDF
- trains a Logistic Regression classifier
- gives ~39% accuracy on the manual evaluation set

Not great, but it shows how noisy weak labels really are.

### DistilBERT Transformer Model

Training happens in two steps:

**1. External pretraining on SST-2**
**2.Fine-tuning on Letterboxd strict labels**

The model:
- tokenises review text
- trains for 3 epochs
- runs on CPU
- saves weights and tokeniser

It reaches ~45% accuracy on the manually labelled set.
Better than the baseline, but still challenged by neutral, sarcastic, and domain-specific language.

## 🧪 Manual Evaluation

Weak labels only go so far. To get a realistic view, a small set of review texts (49 samples) was labeled by hand.

Results:
- Baseline: ~39%
- DistilBERT: ~45%

This difference highlights where transformers help and where weak labels fall short.

## 🌐 FastAPI Service

The API exposes a few routes:

### /health
Quick check that the API is running.

### /analyze-text
Takes raw text and returns:
- predicted label
- probabilities
- which model was used (baseline or transformer)

### /movie-summary
Aggregates sentiment for a specific movie.

### /compare-movies
Compares two movies based on review sentiment.

Run the API using:
`uvicorn api.main:app --reload`

## 🧩 Architecture Overview

```
                     Raw Letterboxd Data
                            |
                            v
                    Data Cleaning Pipeline
                            |
                            v
                -------------------------------
                |                              |
                v                              v
          Baseline Model                 DistilBERT Model
                -------------------------------
                            |
                            v
                      FastAPI Backend
                            |
                            v
                      Clients / Apps

```

## Summary

This project puts together a full sentiment workflow for Letterboxd reviews:

- raw data cleaning
- strict weak-label generation
- baseline (TF-IDF + LR)
- transformer (DistilBERT with SST-2 warm-up)
- a small human-labeled test set
- a FastAPI API for real-time predictions

It shows how models respond to noisy labels, how much transformers help, and how an end-to-end pipeline looks when built from real user-generated text.
