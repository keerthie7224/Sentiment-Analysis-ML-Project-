import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Streamlit title
st.title("🎬 Movie Review Sentiment Analysis")
st.write("Enter a movie review and the model will predict whether it is Positive or Negative.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("IMDB Dataset.csv")
    return df

df = load_data()

# Show dataset info
st.write("Dataset Shape:", df.shape)

# Split data
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# User input
review_input = st.text_area("Enter your movie review")

# Prediction
if st.button("Predict Sentiment"):
    if review_input.strip() != "":
        review_vec = vectorizer.transform([review_input])
        prediction = model.predict(review_vec)

        if prediction[0] == "positive":
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")
    else:
        st.warning("Please enter a review.")