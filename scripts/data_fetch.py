import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start_date, end_date):
    stock = yf.download(ticker, start=start_date, end=end_date)
    stock.reset_index(inplace=True)
    return stock

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2024-01-01", "2024-03-30")
    df.to_csv("../data/raw/AAPL_stock.csv", index=False)
    print(df.head())  # Check first few rows
    print(df.tail())  # Check last few rows
    print(df['Date'].min(), df['Date'].max())  # Confirm date range
    print("Stock data saved!")

# 5️⃣ scripts/news_analysis.py
from textblob import TextBlob

def analyze_sentiment(news_text):
    return TextBlob(news_text).sentiment.polarity

if __name__ == "__main__":
    sentiment = analyze_sentiment("Apple's stock is performing exceptionally well!")
    print(f"Sentiment Score: {sentiment}")