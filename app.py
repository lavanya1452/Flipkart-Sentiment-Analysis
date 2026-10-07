import joblib
import pandas as pd
import streamlit as st
from numbers import Integral

st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer", page_icon="🛍️", layout="centered"
)


def sentiment_name(label):
    if isinstance(label, Integral):
        return {0: "Negative", 1: "Neutral", 2: "Positive"}.get(int(label), str(label))
    return str(label)


@st.cache_resource
def load_assets():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


try:
    model, vectorizer = load_assets()

    st.title("🛍️ Flipkart Review Sentiment Analysis")
    st.write(
        "Enter a customer review below to classify its sentiment (Positive, Neutral, Negative)."
    )

    review_input = st.text_area(
        "Enter Review Text:",
        placeholder="e.g., the product was very damaged",
        height=150,
    )

    if st.button("Analyze Sentiment", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a valid review text.")
        else:
            # Transform text directly
            vec_input = vectorizer.transform([review_input])

            prediction = model.predict(vec_input)[0]
            probabilities = model.predict_proba(vec_input)[0]
            classes = model.classes_
            class_names = [sentiment_name(label) for label in classes]
            predicted_sentiment = sentiment_name(prediction)

            confidence_map = dict(zip(class_names, probabilities))
            confidence = confidence_map[predicted_sentiment] * 100

            st.markdown("---")
            if predicted_sentiment == "Positive":
                st.subheader("Predicted Sentiment: 🟢 Positive")
            elif predicted_sentiment == "Neutral":
                st.subheader("Predicted Sentiment: 🟡 Neutral")
            else:
                st.subheader("Predicted Sentiment: 🔴 Negative")

            st.write(f"**Confidence Score:** `{confidence:.2f}%`")

            # Breakdown Table
            prob_df = pd.DataFrame(
                {
                    "Sentiment": class_names,
                    "Probability (%)": [p * 100 for p in probabilities],
                }
            )
            st.dataframe(prob_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading model assets: {e}")