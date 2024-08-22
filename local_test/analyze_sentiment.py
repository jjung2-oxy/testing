from transformers import pipeline

# Specify a model to avoid the warning
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiments(relevant_sentences):
    sentiments = analyze_sentiment(relevant_sentences)
    print(f"Debug: Analyzed sentiment for {len(sentiments)} sentences")
    return sentiments

def analyze_sentiment(sentences):
    try:
        sentiments = sentiment_pipeline(sentences)
        return sentiments
    except Exception as e:
        print(f"Debug: Error in sentiment analysis - {str(e)}")
        return []