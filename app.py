from pathlib import Path
import pickle

import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "sentiment_model.pkl"
VECTORIZER_PATH = ROOT / "tfidf_vectorizer.pkl"

st.set_page_config(page_title="Flipkart Sentiment Analyzer", page_icon="🛒")


@st.cache_resource
def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


try:
    model, vectorizer = load_artifacts()
except FileNotFoundError:
    st.error("Model files are missing. Run train_model.py after adding the dataset CSV.")
    st.stop()

st.title("Flipkart Review Sentiment Analyzer")
review = st.text_area("Customer review", placeholder="Type or paste a review")

if st.button("Analyze sentiment", type="primary"):
    if not review.strip():
        st.warning("Enter a review to analyze.")
    else:
        features = vectorizer.transform([review])
        sentiment = model.predict(features)[0]
        confidence = float(model.predict_proba(features).max())
        st.subheader(str(sentiment))
        st.write(f"Confidence: {confidence:.1%}")