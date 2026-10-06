from pathlib import Path
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Flipkart_Reviews_Sentiment_Analysis_30000x25.csv"
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.pkl"


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    dataset = pd.read_csv(DATA_PATH)
    required_columns = {"Review_Summary", "Review_Text", "Sentiment"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    dataset = dataset.dropna(subset=["Sentiment"]).copy()
    reviews = (
        dataset["Review_Summary"].fillna("").astype(str)
        + " "
        + dataset["Review_Text"].fillna("").astype(str)
    ).str.strip()
    labels = dataset["Sentiment"].astype(str)

    if len(dataset) < 2 or labels.nunique() < 2:
        raise ValueError("At least two labeled sentiment classes are required to train.")

    stratify = labels if labels.value_counts().min() >= 2 else None
    train_reviews, test_reviews, train_labels, test_labels = train_test_split(
        reviews,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    train_features = vectorizer.fit_transform(train_reviews)
    test_features = vectorizer.transform(test_reviews)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train_features, train_labels)
    predictions = model.predict(test_features)

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

    print(f"Accuracy: {accuracy_score(test_labels, predictions):.4f}")
    print(classification_report(test_labels, predictions, zero_division=0))
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()