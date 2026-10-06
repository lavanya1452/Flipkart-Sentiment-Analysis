import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Flipkart Review Sentiment Analyzer", page_icon="🛍️", layout="centered"
)


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

            confidence_map = dict(zip(classes, probabilities))
            confidence = confidence_map[prediction] * 100

            st.markdown("---")
            if prediction == "Positive":
                st.subheader("Predicted Sentiment: 🟢 Positive")
            elif prediction == "Neutral":
                st.subheader("Predicted Sentiment: 🟡 Neutral")
            else:
                st.subheader("Predicted Sentiment: 🔴 Negative")

            st.write(f"**Confidence Score:** `{confidence:.2f}%`")

            # Breakdown Table
            prob_df = pd.DataFrame(
                {
                    "Sentiment": classes,
                    "Probability (%)": [p * 100 for p in probabilities],
                }
            )
            st.dataframe(prob_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading model assets: {e}")