# NLP Movie Review Chatbot

A simple yet powerful intent-based chatbot built with Python and NLP. This project combines a conversational AI (for NLP and AI topics) with a machine-learning-based Movie Review Sentiment Analyzer trained on the Kaggle *Sentiment Analysis on Movie Reviews* dataset.

## 🚀 Features

- **Intent-Based Interaction**: Responds to greetings and answers technical questions about NLP and AI.
- **NLP Preprocessing**: Custom pipeline involving tokenization, lemmatization, and stop-word removal using `NLTK`.
- **Sentiment Analysis**: A Logistic Regression model trained to classify movie reviews into five sentiment categories (Negative to Positive).
- **Graceful Fallback**: If the Kaggle dataset is not found, the system automatically falls back to the NLTK `movie_reviews` corpus.

---

## 🛠️ Local Setup

Follow these steps to get the project running on your local machine:

### 1. Initialize Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install scikit-learn nltk pandas numpy kaggle
```

### 3. Setup Project Data
Ensure your `train.tsv` and `test.tsv` files are located in the `sentiment-analysis-on-movie-reviews/` folder. (The script will automatically unzip them if they are in `.zip` format).

---

## 🏃 How to Run

1. **Launch the Chatbot**:
   ```bash
   cd sentiment-analysis-on-movie-reviews
   python3 chatbot.py
   ```
2. **Initialization**: The bot will take a few seconds to train the sentiment model upon startup. Once you see "Bot: Hello! I'm ready.", you can start chatting.

---

## 💬 Sample Questions to Ask

### NLP & AI Concepts
- "What is Natural Language Processing?"
- "Explain the concept of Lemmatization."
- "What is Named Entity Recognition (NER)?"
- "How is AI used in the healthcare industry?"
- "What are some real-world applications of AI?"

### Movie Review Sentiment Analysis
To trigger a sentiment check, use keywords like **"analyze"**, **"movie"**, or **"sentiment"**.
- **Bot**: "Please enter the movie review text:"
- **Try these reviews**:
  - *"This movie was an absolute masterpiece! The acting was incredible."* (**Positive**)
  - *"I found the plot boring and the characters were very poorly developed."* (**Negative**)
  - *"It was an okay film, some parts were good but others were quite slow."* (**Neutral**)

---

## 📂 Project Structure
- `chatbot.py`: Main entry point and conversational logic.
- `model.py`: Sentiment analysis training and prediction logic.
- `preprocessing.py`: NLP text cleaning utilities.
- `intents.json`: Definition of chatbot intents and responses.
- `.gitignore`: Configured to exclude `venv`, data files, and system artifacts.
