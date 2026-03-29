import json
import random
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import NLPPreprocessor
from model import SentimentAnalyzer, load_data

class IntentChatbot:
    def __init__(self, intents_file):
        with open(intents_file, 'r') as f:
            self.intents = json.load(f)['intents']
        self.preprocessor = NLPPreprocessor()
        self.vectorizer = TfidfVectorizer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.sentiment_model_ready = False
        self.is_kaggle = False
        self.prepare_data()

    def prepare_data(self):
        self.patterns = []
        self.tag_map = []
        for intent in self.intents:
            for pattern in intent['patterns']:
                clean_p = self.preprocessor.clean_text(pattern)
                self.patterns.append(clean_p)
                self.tag_map.append(intent['tag'])
        self.tfidf_matrix = self.vectorizer.fit_transform(self.patterns)

    def train_sentiment(self):
        X, y, is_kaggle = load_data()
        self.sentiment_analyzer.train(X, y)
        self.sentiment_model_ready = True
        self.is_kaggle = is_kaggle

    def get_response(self, user_input):
        clean_input = self.preprocessor.clean_text(user_input)
        if not clean_input:
            return "I'm sorry, I didn't quite catch that. Could you please rephrase?"

        input_vector = self.vectorizer.transform([clean_input])
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)
        max_idx = np.argmax(similarities)
        max_sim = similarities[0][max_idx]

        if max_sim < 0.2:
            return "I'm not sure I understand. I can help with NLP concepts, AI applications, or movie review sentiment analysis."

        tag = self.tag_map[max_idx]
        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return "I'm not sure how to respond to that."

def main():
    print("Bot: Initializing (this may take a moment)...")
    chatbot = IntentChatbot('intents.json')
    chatbot.train_sentiment()
    
    print("\nBot: Hello! I'm ready. Type 'bye' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['bye', 'exit', 'quit']:
            print(f"Bot: {chatbot.get_response('bye')}")
            break
        
        if "sentiment" in user_input.lower() or "analyze" in user_input.lower() or "movie" in user_input.lower():
            review = input("Bot: Please enter the movie review text: ")
            prediction = chatbot.sentiment_analyzer.predict(review)
            
            if chatbot.is_kaggle:
                # Kaggle labels: 0, 1, 2, 3, 4
                labels_map = {0: "Negative", 1: "Somewhat Negative", 2: "Neutral", 3: "Somewhat Positive", 4: "Positive"}
                result = labels_map[prediction]
            else:
                # NLTK labels: 0, 1
                result = "Positive" if prediction == 1 else "Negative"
                
            print(f"Bot: The sentiment of that review is {result}.")
            continue

        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
