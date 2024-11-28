st.set_page_config(page_title="Book Genre Prediction App")

import streamlit as st
import pandas as pd
import joblib
import re
import nltk

# Load pre-trained model, vectorizer, and label encoder
svc = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
LE = joblib.load('label_encoder.pkl')

# Download NLTK resources if not available
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK resources
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

stop_words = set(stopwords.words('english'))
lemma = WordNetLemmatizer()
stemmer = PorterStemmer()

# Text preprocessing pipeline
def clean_text(text):
    text = re.sub("\'", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    return ' '.join(text.split()).lower()

def remove_stopwords(text, stop_words):
    return ' '.join([w for w in text.split() if w not in stop_words])

def lemmatizing(text, lemma):
    return ' '.join([lemma.lemmatize(word) for word in text.split()])

def stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])

def preprocess_text(text, stop_words, lemma, stemmer):
    text = clean_text(text)
    text = remove_stopwords(text, stop_words)
    text = lemmatizing(text, lemma)
    text = stemming(text, stemmer)
    return text

# Streamlit app interface
st.title("Book Genre Prediction")
st.write("Enter a book title below to predict its genre.")

# Load dataset (optional for reference)
books = pd.read_csv('BooksDataSet.csv')

# Input form
book_name = st.text_input("Book Title")

if st.button("Predict Genre"):
    if book_name:
        # Try to find the book's summary
        book_summary = books[books['book_name'].str.contains(book_name, case=False, na=False)]['summary']

        if not book_summary.empty:
            book_summary = book_summary.iloc[0]
            processed_summary = preprocess_text(book_summary, stop_words, lemma, stemmer)
        else:
            st.write("Book not found in dataset. Predicting based on title itself...")
            processed_summary = preprocess_text(book_name, stop_words, lemma, stemmer)

        # Transform and predict
        book_vec = tfidf_vectorizer.transform([processed_summary])
        predicted_genre = svc.predict(book_vec)
        predicted_genre_label = LE.inverse_transform(predicted_genre)[0]
        st.success(f"Predicted genre: **{predicted_genre_label}**")
    else:
        st.error("Please enter a valid book title!")
