import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class NLPPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stop_words]
        return " ".join(tokens)

if __name__ == "__main__":
    preprocessor = NLPPreprocessor()
    print("Preprocessed: " + preprocessor.clean_text("I love NLP!"))
