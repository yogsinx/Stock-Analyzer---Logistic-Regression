from textblob import TextBlob

def analyze_sentiment(news_text):
    return TextBlob(news_text).sentiment.polarity

if __name__ == "__main__":
    sentiment = analyze_sentiment("Apple's stock is performing exceptionally well!")
    print(f"Sentiment Score: {sentiment}")