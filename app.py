from pathlib import Path

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent

# Configure Streamlit page
st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer", page_icon="🛍️", layout="centered"
)


# Load artifacts with caching
@st.cache_resource
def load_assets():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    model = joblib.load(ROOT / "sentiment_model.pkl")
    vectorizer = joblib.load(ROOT / "tfidf_vectorizer.pkl")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return model, vectorizer, stop_words, lemmatizer


def clean_text(text, stop_words, lemmatizer):
    words = [
        lemmatizer.lemmatize(word)
        for word in str(text).lower().split()
        if word.isalpha() and word not in stop_words
    ]
    return " ".join(words)


try:
    model, vectorizer, stop_words, lemmatizer = load_assets()
except FileNotFoundError:
    st.error("Model files are missing. Run train_model.py after adding the dataset CSV.")
    st.stop()

st.title("🛍️ Flipkart Review Sentiment Analysis")
st.write(
    "Enter a customer review below to classify its sentiment (Positive, Neutral, Negative)."
)

review_input = st.text_area(
    "Enter Review Text:",
    placeholder="e.g., The product arrived damaged and customer support was completely unhelpful.",
    height=150,
)

if st.button("Analyze Sentiment", type="primary"):
    if not review_input.strip():
        st.warning("Please enter a valid review text.")
    else:
        cleaned_review = clean_text(review_input, stop_words, lemmatizer)
        vec_input = vectorizer.transform([cleaned_review])

        prediction = model.predict(vec_input)[0]
        probabilities = model.predict_proba(vec_input)[0]
        classes = model.classes_

        confidence_map = dict(zip(classes, probabilities))
        confidence = confidence_map[prediction] * 100

        color_map = {
            "Positive": "🟢 Positive",
            "Neutral": "🟡 Neutral",
            "Negative": "🔴 Negative",
        }

        st.markdown("---")
        st.subheader(
            f"Predicted Sentiment: {color_map.get(prediction, prediction)}"
        )
        st.write(f"**Confidence Score:** `{confidence:.2f}%`")

        st.write("#### Confidence Breakdown:")
        prob_df = pd.DataFrame(
            {
                "Sentiment": classes,
                "Probability (%)": [p * 100 for p in probabilities],
            }
        )
        st.dataframe(prob_df, use_container_width=True)