from transformers import pipeline
from promptflow import tool

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@tool
def analyze_sentiment(sentences: list) -> list:
    sentiments = sentiment_pipeline(sentences)
    return sentiments