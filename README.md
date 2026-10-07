# 🛍️ Flipkart Reviews Sentiment Analysis System

An end-to-end Machine Learning & Natural Language Processing (NLP) system built to classify customer review sentiments into **Positive**, **Neutral**, or **Negative** categories. This repository includes a complete data processing pipeline, model training with class imbalance handling, and an interactive Streamlit web application deployed on Streamlit Community Cloud.

---

## 📌 Project Architecture

```text
Flipkart-Sentiment-Analysis/
├── Flipkart_Reviews_Sentiment_Analysis_30000x25.csv   # Dataset
├── Flipkart_Sentiment_Analysis.ipynb                  # Exploratory analysis & experiments notebook
├── train_model.py                                     # Model training & asset export script
├── app.py                                             # Interactive Streamlit Web Application
├── requirements.txt                                   # Dependencies
├── .python-version                                    # Python version pinning (3.11)
├── sentiment_model.pkl                                # Trained Logistic Regression Model
└── tfidf_vectorizer.pkl                               # TF-IDF Vectorizer Artifact
