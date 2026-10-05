import streamlit as st
import tensorflow as tf
import pickle
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
nltk.download("stopwords", quiet=True)

# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# -----------------------------
# Load model and preprocessing
# -----------------------------

@st.cache_resource
def load_model_and_tools():

    model = tf.keras.models.load_model(
        "models/sentiment_lstm.keras"
    )

    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("models/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_model_and_tools()

# -----------------------------
# Text preprocessing
# -----------------------------

stop_words = set(stopwords.words("english"))


def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def remove_stopwords(text):

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


# -----------------------------
# Prediction
# -----------------------------

def predict_sentiment(review):

    cleaned = clean_text(review)
    cleaned = remove_stopwords(cleaned)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = pad_sequences(
        sequence,
        maxlen=100,
        padding="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )[0]

    index = prediction.argmax()

    sentiment = label_encoder.inverse_transform(
        [index]
    )[0]

    confidence = prediction[index] * 100

    return sentiment, confidence


# -----------------------------
# UI
# -----------------------------

st.title("🛒 Flipkart Review Sentiment Analyzer")

st.write(
    "Enter a customer review and the LSTM model "
    "will predict its sentiment."
)

review = st.text_area(
    "Enter your review:",
    placeholder="Example: The product quality is excellent and delivery was very fast!"
)

if st.button("🔍 Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:

        sentiment, confidence = predict_sentiment(review)

        st.subheader("Prediction")

        if sentiment == "Positive":
            st.success(f"😊 Positive — {confidence:.2f}% confidence")

        elif sentiment == "Negative":
            st.error(f"😞 Negative — {confidence:.2f}% confidence")

        else:
            st.warning(f"😐 Neutral — {confidence:.2f}% confidence")

        st.write(
            f"**Confidence:** {confidence:.2f}%"
        )
