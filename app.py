
# --- Imports ---
import yfinance as yf
import praw
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.optimizers import Adam
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env.local')

# --- Reddit API Setup ---
reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent="REDDIT_USER_AGENT"
)

# --- FinBERT Sentiment Model ---
finbert_model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
finbert_tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["positive", "neutral", "negative"]
    return labels[probs.argmax().item()]

def fetch_reddit_comments(stock_ticker, limit=1000):
    print("[Process] Searching Reddit for relevant comments...")
    query = f'(title:"{stock_ticker}" OR selftext:"{stock_ticker}") AND (title:stock OR title:share OR title:earnings OR title:market)'
    finance_keywords = [
        "stock", "share", "ticker", "dividend", "price", "earnings",
        "revenue", "guidance", "options", "puts", "calls", "market",
        "invest", "investing", "hold", "buy", "sell", "portfolio"
    ]
    subreddit = reddit.subreddit("all")
    records = []
    def is_finance_related(text, keywords):
        text_lower = text.lower()
        return any(word in text_lower for word in keywords)
    for submission in subreddit.search(query, sort="relevance", time_filter="month", limit=limit):
        submission.comments.replace_more(limit=0)
        for comment in submission.comments.list():
            clean_body = comment.body.replace("\n", " ").strip()
            if (
                is_finance_related(clean_body, finance_keywords) and
                len(clean_body.split()) > 5 and
                comment.author not in ["AutoModerator", None]
            ):
                records.append({
                    "Post Title": submission.title,
                    "Post ID": submission.id,
                    "Comment ID": comment.id,
                    "Comment": clean_body,
                    "Score": comment.score,
                    "Author": str(comment.author),
                    "Created UTC": pd.to_datetime(comment.created_utc, unit='s')
                })
    print(f"[Process] {len(records)} Reddit comments fetched.")
    df_comments = pd.DataFrame(records)
    if not df_comments.empty:
        print("[Process] Analyzing sentiment for Reddit comments...")
        df_comments.dropna(subset=["Comment"], inplace=True)
        df_comments["Sentiment"] = df_comments["Comment"].apply(lambda x: analyze_sentiment(x))
        df_comments["date"] = df_comments["Created UTC"].dt.date
        print("[Process] Sentiment analysis complete.")
    return df_comments

def fetch_stock_data(stock_ticker, period="5y"):
    print("[Process] Downloading stock data from Yahoo Finance...")
    df = yf.download(stock_ticker, period=period, timeout=30)
    df = df[["Close", "Volume"]]
    df["Return"] = df["Close"].pct_change()
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={'Date': 'date'})
    df_reset['date'] = pd.to_datetime(df_reset['date'])
    df_reset.columns = df_reset.columns.get_level_values(-2)
    print("[Process] Stock data ready.")
    return df_reset

def prepare_merged(stock_ticker):
    print(f"[Process] Fetching stock data for {stock_ticker}...")
    stock_df = fetch_stock_data(stock_ticker)
    print(f"[Process] Fetching Reddit comments and sentiment for {stock_ticker}...")
    comments_df = fetch_reddit_comments(stock_ticker)
    if comments_df.empty:
        print("[Process] No relevant Reddit comments found. Proceeding with stock data only.")
        # Create dummy sentiment columns
        stock_df["positive"] = 0
        stock_df["neutral"] = 1
        stock_df["negative"] = 0
        return stock_df
    print("[Process] Aggregating sentiment by date...")
    sentiment = comments_df.groupby("date")["Sentiment"].value_counts().unstack()
    for col in ["positive", "neutral", "negative"]:
        if col not in sentiment:
            sentiment[col] = 0
    sentiment = sentiment[["positive", "neutral", "negative"]].div(sentiment.sum(axis=1), axis=0)
    sentiment = sentiment.reindex(stock_df["date"].dt.date, fill_value=0)
    sentiment.reset_index(inplace=True)
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment.reset_index(inplace=True)
    merged = pd.merge(stock_df, sentiment, on="date", how="inner")
    merged.fillna(0, inplace=True)
    print("[Process] Data merge complete.")
    return merged

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def run_analysis():
    while True:
        stock_ticker = input("\nEnter stock ticker (or 'quit' to exit): ").upper().strip()
        if stock_ticker == "QUIT":
            print("Exiting program.")
            break
        try:
            print("[Process] Preparing merged dataset...")
            merged = prepare_merged(stock_ticker)
            features = ["Close", "Volume", "Return", "positive", "neutral", "negative"]
            target = "Return"
            print("[Process] Scaling features...")
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(merged[features])
            X, y = [], []
            window = 30
            print("[Process] Creating time series windows...")
            for i in range(window, len(scaled)):
                X.append(scaled[i-window:i])
                y.append(scaled[i, features.index(target)])
            X, y = np.array(X), np.array(y)
            if len(X) == 0:
                print("[Process] Not enough data to train LSTM. Try a different ticker or longer period.")
                continue
            print(f"[Process] Training LSTM model on {X.shape[0]} samples...")
            model = build_lstm_model((X.shape[1], X.shape[2]))
            model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, shuffle=False, verbose=1)
            print("[Process] Predicting next-day return...")
            last_window = X[-1:]
            pred_return = model.predict(last_window)[0][0]
            last_close = merged["Close"].iloc[-1]
            pred_close = last_close * (1 + pred_return)
            print(f"\n✅ Analysis complete for {stock_ticker}:")
            print(f"- Latest Close Price: {last_close:.2f}")
            print(f"- Predicted Next Close Price: {pred_close:.2f}")
        except Exception as e:
            print(f"⚠️ Error analyzing {stock_ticker}: {e}")

if __name__ == "__main__":
    run_analysis()
