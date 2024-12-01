
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import random

st.set_page_config(page_title="BookWise")

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
st.markdown("<h1 style='text-align: center; color: #000000;'>📚 BookWise</h1>", unsafe_allow_html=True)
st.write("Enter a book title below to predict its genre.")

# Load dataset (optional for reference)
books = pd.read_csv('BooksDataSet.csv')

# Input form
book_name = st.text_input("Book Title")

# Function to recommend books by genre with randomization
def recommend_books_by_genre(input_title, books):
    # Try to find the book's genre based on the summary
    book_summary = books[books['book_name'].str.contains(input_title, case=False, na=False)]['summary']
    
    if not book_summary.empty:
        # If the book's summary is found, use it to find genre and recommend
        book_summary = book_summary.iloc[0]
        processed_summary = preprocess_text(book_summary, stop_words, lemma, stemmer)
        genre = books[books['summary'] == book_summary]['genre'].iloc[0]  # Genre of the matched book
        
        # Filter books by the same genre and shuffle the recommendations
        similar_books = books[books['genre'] == genre].sample(n=5, random_state=random.randint(0, 1000))  # Recommend 5 books randomly
        return similar_books
    else:
        # If no summary match, use the title to predict genre and recommend
        processed_title = preprocess_text(input_title, stop_words, lemma, stemmer)
        book_vec = tfidf_vectorizer.transform([processed_title])
        predicted_genre = svc.predict(book_vec)
        predicted_genre_label = LE.inverse_transform(predicted_genre)[0]
        
        # Filter books by the predicted genre and shuffle the recommendations
        similar_books = books[books['genre'] == predicted_genre_label].sample(n=5, random_state=random.randint(0, 1000))  # Recommend 5 books randomly
        return similar_books

# Handle book genre prediction and recommendation
if st.button("Predict Genre"):
    if book_name:
        # Try to find the book's summary
        book_summary = books[books['book_name'].str.contains(book_name, case=False, na=False)]['summary']

        if not book_summary.empty:
            book_summary = book_summary.iloc[0]
            processed_summary = preprocess_text(book_summary, stop_words, lemma, stemmer)
        else:
            processed_summary = preprocess_text(book_name, stop_words, lemma, stemmer)

        # Transform and predict
        book_vec = tfidf_vectorizer.transform([processed_summary])
        predicted_genre = svc.predict(book_vec)
        predicted_genre_label = LE.inverse_transform(predicted_genre)[0]
        with st.spinner('Predicting genre...'):
            st.success(f"Predicted genre: **{predicted_genre_label}**")
    else:
        st.error("Please enter a valid book title!")


if st.button("Recommend Similar Books"):
    if book_name:
        similar_books = recommend_books_by_genre(book_name, books)
        if not similar_books.empty:
            st.write("Similar Books you might like")
            for index, row in similar_books.iterrows():
                st.write(f"• **{row['book_name']}** ")
        else:
            st.write("Sorry, no similar books found based on the provided title.")
    else:
        st.error("Please enter a valid book title!")

with st.expander("📖 About the App"):
    st.write("""
        This app predicts the genre of a book based on its title or summary. It uses a trained machine learning model 
        to classify the genre and provide an accurate prediction. You can also get recommendations of similar books
        based on the same genre.""")

