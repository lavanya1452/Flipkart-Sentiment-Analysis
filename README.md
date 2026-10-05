# Flipkart Reviews Sentiment Analysis

An end-to-end Natural Language Processing (NLP), Machine Learning and Deep Learning project for classifying Flipkart-style customer reviews into Positive, Neutral and Negative sentiment.

## Project Overview

E-commerce platforms receive large volumes of customer reviews. Manually analyzing these reviews is time-consuming.

This project develops an automated sentiment analysis system that processes customer reviews and predicts their sentiment.

The project covers:

- Data preprocessing and cleaning
- Exploratory Data Analysis
- NLP text preprocessing
- TF-IDF feature extraction
- Traditional Machine Learning
- Deep Learning
- Model evaluation
- Sentiment prediction for new reviews

## Dataset

The project uses a synthetic Flipkart-style review dataset containing:

- 30,000 records
- 25 columns
- Review text and review summary
- Product and customer information
- Rating and delivery information
- Sentiment labels

### Target Classes

- Positive
- Neutral
- Negative

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- TensorFlow / Keras

## Machine Learning Models

The project compares:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM

## Deep Learning Models

The project investigates:

1. Embedding + Dense Neural Network
2. LSTM
3. Bidirectional LSTM

## NLP Pipeline

```text
Raw Review
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Stopword Removal
    ↓
TF-IDF / Sequence Representation
    ↓
Machine Learning / Deep Learning
    ↓
Sentiment Prediction
