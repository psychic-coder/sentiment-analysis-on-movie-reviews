# NLP Movie Review Chatbot

Simple intent-based chatbot for NLP concepts and AI applications, with movie review sentiment analysis.

## Setup
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install scikit-learn nltk pandas numpy kaggle`
4. `cd sentiment-analysis-on-movie-reviews`
5. `python3 chatbot.py`

## Project Structure
All components (Intents, NLP Preprocessing, Sentiment Model, and Chatbot) are located in the `sentiment-analysis-on-movie-reviews/` directory. The model will automatically train on the `train.tsv` file provided.
