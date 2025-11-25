# 🎬 Letterboxd Movie Sentiment API – Baseline, DistilBERT, Weak Labels, Manual Evaluation, FastAPI

##  📌 Overview

This project builds a sentiment analysis system for Letterboxd movie reviews. Letterboxd reviews are messy, sarcastic, emotional, and often don’t match the star rating.
The goal is to build a full NLP pipeline that predicts whether a movie review is positive or negative, even when labels are noisy.

People review movies in many different ways, and simple rules aren’t enough.
A good sentiment model should understand:
- how a review is written
- what opinion the user expresses
- how to deal with slang, jokes, and sarcasm
- how to handle weak labels like star ratings

This project brings all these ideas together.
It uses a scraped Letterboxd dataset and includes:
- a data cleaning pipeline
- strict weak-label generation
- a baseline TF-IDF + Logistic Regression model
- a transformer model (DistilBERT)
- external pretraining on SST-2
- a manual evaluation set
- a FastAPI backend

## 🗂️ Data

Source: A CSV of scraped Letterboxd reviews.

Fields include:
- Movie name
- Release year
- Review text
- Reviewer name
- Broken Unicode star ratings
- Review date
- Like and comment counts

After cleaning, the dataset produces:
- numeric star ratings
- clean review text
- loose 3-class sentiment (pos/neu/neg)
- strict binary sentiment (pos/neg)
- ~3500 total reviews
- ~514 strict weak-labeled reviews
- 49 human-labeled samples for real evaluation

All processed files are saved into `data/processed/`.

## 🔧 Data Pipeline (src/preprocess + src/prepare_data)

The pipeline handles all preparation stages:
- Load the raw reviews
- Fix corrupted Unicode star ratings
- Convert stars to numeric ratings
- Create strict binary labels
- Clean the review text
- Parse dates
- Filter rows with missing review text
- Save the cleaned dataset into `data/processed/`

This keeps everything reproducible and keeps notebooks and training scripts clean.

## 🔎 Notebooks Breakdown
### 1. EDA

Explores the cleaned dataset:
- rating distribution
- strict label distribution
- review length patterns
- check for corrupted characters
- sample review inspection

### 2. Baseline Model Training

Runs a simple baseline:
- loads strict binary labels
- filters very short reviews
- balances classes
- trains TF-IDF + Logistic Regression
- prints validation accuracy
- saves model and vectorizer

This acts as the baseline sentiment classifier.

## 🤖 Baseline and Transformer Models
### Baseline: TF-IDF + Logistic Regression

The baseline code:
- loads the cleaned dataset
- filters by strict labels
- balances positive and negative
- vectorizes text
- trains Logistic Regression
- evaluates on a validation split
- exposes a predict function for the API

Baseline accuracy on human-labeled samples is about 39%.

### DistilBERT Transformer Model

The transformer training includes two stages:

**Stage 1: External Pretraining (SST-2)**
Improves the model by teaching it real sentiment before applying it to noisy Letterboxd data.

**Stage 2: Fine-tuning on Letterboxd**
Uses strict labels to adapt the model to the domain.

**The model:**
- tokenizes the reviews
- trains for 3 epochs
- runs on CPU
- saves model + tokenizer into models/distilbert/
- integrates with the API

DistilBERT accuracy on human-labeled samples is about 45%, better than the baseline.

## 🧪 Manual Evaluation

Since star ratings are weak labels, the project includes a small gold-standard test set.

**Steps:**
- Sample 150 strict-label reviews
- Manually label 50 reviews with real sentiment
- Evaluate both models on these true labels

**Results:**
- Baseline: ~39% accuracy
- DistilBERT: ~45% accuracy

Neutral reviews are hard because the models were trained only on positive and negative, but this highlights the difference between weak labels and true sentiment.

## 🌐 FastAPI Service

The FastAPI backend exposes:

### /health
Quick check that the API is running.

### /analyze-text
Takes raw text and returns:
- predicted label
- probabilities
- which model was used

You can switch between:
`"model_type": "baseline"`
and
`"model_type": "transformer"`

### /movie-summary
Summarizes a movie using all reviews:
- count of positive and negative
- average sentiment
- rating overview

### /compare-movies
Compares two movies using analytics functions.

Run the API using:
`uvicorn api.main:app --reload`

## 🧩 Architecture Overview

```
                     Raw Letterboxd Data
                             |
                             v
                   Data Cleaning Pipeline
               - rating fix
               - text cleaning
               - strict labels
                             |
                             v
           ---------------------------------------
           |                                     |
           v                                     v
   Baseline Model                        DistilBERT Transformer
   - TF-IDF                              - SST-2 pretraining
   - Logistic Regression                 - LBX fine-tuning
   - 39% on manual eval                  - 45% on manual eval
           |                                     |
           ------------------   -------------------
                              v
                        FastAPI Backend
                    /analyze-text (baseline/bert)
                    /movie-summary
                    /compare-movies
                              |
                              v
                       Clients / Apps
```

## Summary

This project builds a complete sentiment analysis system for Letterboxd reviews using both a TF-IDF baseline and a DistilBERT transformer. It handles noisy rating-based labels with strict filtering and includes a small human-labeled set for real evaluation. Everything is served through a FastAPI backend with endpoints for text analysis, movie summaries, and comparisons.
