import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib

# Download NLTK resources
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

# Load dataset
books = pd.read_csv('BooksDataSet.csv')
books = books[['book_id', 'book_name', 'genre', 'summary']]

# Preprocess summaries
books['processed_summary'] = books['summary'].apply(lambda x: preprocess_text(x, stop_words, lemma, stemmer))

# Encode genres
LE = LabelEncoder()
y = LE.fit_transform(books['genre'])

# Split the data
xtrain, xval, ytrain, yval = train_test_split(books['processed_summary'], y, test_size=0.2, random_state=557)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xval)

# Train SVM model
svc = SVC(kernel='rbf', gamma=1)
svc.fit(xtrain_tfidf, ytrain)

# Save the trained model and vectorizer
joblib.dump(svc, 'svm_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(LE, 'label_encoder.pkl')

print("Model training complete. Files saved: `svm_model.pkl`, `tfidf_vectorizer.pkl`, and `label_encoder.pkl`.")
