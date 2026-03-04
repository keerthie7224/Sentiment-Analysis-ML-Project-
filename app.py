import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset
data = {
    "review": [
        "This product is amazing",
        "I love this phone",
        "Worst product ever",
        "Very bad quality",
        "Excellent service",
        "Not worth the money",
        "Highly recommended",
        "Terrible experience"
    ],
    "sentiment": [
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Positive",
        "Negative",
        "Positive",
        "Negative"
    ]
}

df = pd.DataFrame(data)

# Features and labels
X = df["review"]
y = df["sentiment"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a review to check whether it is Positive or Negative.")

# User input
user_input = st.text_area("Enter your review")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)

        st.subheader("Prediction:")
        st.success(prediction[0])
    else:
        st.warning("Please enter a review.")