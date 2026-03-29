import os
import pandas as pd
import numpy as np
import zipfile
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import NLPPreprocessor

def unzip_data():
    for file in os.listdir('.'):
        if file.endswith('.zip'):
            print(f"Unzipping {file}...")
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall('.')
            print(f"Extracted {file}.")

def load_data():
    # Ensure data is unzipped
    if not os.path.exists('train.tsv') and os.path.exists('train.tsv.zip'):
        unzip_data()
    
    if os.path.exists('train.tsv'):
        print("Loading Kaggle Movie Review dataset (TSV)...")
        df = pd.read_csv('train.tsv', sep='\t')
        # Kaggle dataset has 'Phrase' and 'Sentiment'
        return df['Phrase'].tolist(), df['Sentiment'].tolist(), True
    
    # Fallback to NLTK movie_reviews
    print("Kaggle dataset not found. Falling back to NLTK movie_reviews corpus...")
    from nltk.corpus import movie_reviews
    nltk.download('movie_reviews', quiet=True)
    
    documents = []
    labels = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append(movie_reviews.raw(fileid))
            labels.append(1 if category == 'pos' else 0)
    
    return documents, labels, False

class SentimentAnalyzer:
    def __init__(self):
        self.preprocessor = NLPPreprocessor()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        print(f"Preprocessing and shuffling {len(X)} samples... this may take a moment.")
        
        # Convert to numpy for better indexing/shuffling
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle the data to get a better distribution of labels
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        # Use a larger subset for better accuracy (100k samples)
        sample_size = min(100000, len(X))
        X_subset = X[:sample_size]
        y_subset = y[:sample_size]
        
        print(f"Cleaning {sample_size} phrases...")
        X_clean = [self.preprocessor.clean_text(text) for text in X_subset]
        
        print("Vectorizing data...")
        X_tfidf = self.vectorizer.fit_transform(X_clean)
        
        print("Training model...")
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_subset, test_size=0.1, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        print(f"Training Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    def predict(self, text):
        clean_text = self.preprocessor.clean_text(text)
        tfidf = self.vectorizer.transform([clean_text])
        return self.model.predict(tfidf)[0]

def main():
    X, y, is_kaggle = load_data()
    analyzer = SentimentAnalyzer()
    analyzer.train(X, y)
    
    test_review = "This movie was absolutely amazing! I loved every minute of it."
    sentiment = analyzer.predict(test_review)
    print(f"Prediction for '{test_review}': {sentiment}")

if __name__ == "__main__":
    main()
