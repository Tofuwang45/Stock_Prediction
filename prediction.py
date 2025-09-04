import pandas as pd
import yfinance as yf
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import praw

# ============ DATA FETCHING & CLEANING (unchanged) ============
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


def fetch_stock_data(ticker, period="1y"):
    print("[Process] Downloading stock data from Yahoo Finance...")
    df = yf.download(ticker, period=period, timeout=30)
    df = df[["Close", "Volume"]]
    df["Return"] = df["Close"].pct_change()
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={'Date': 'date'})
    df_reset['date'] = pd.to_datetime(df_reset['date'])
    df_reset.columns = df_reset.columns.get_level_values(-2)
    print("[Process] Stock data ready.")
    return df_reset


def analyze_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return "neutral"
    inputs = finbert_tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    outputs = finbert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["positive", "neutral", "negative"]
    return labels[probs.argmax().item()]


def fetch_reddit_comments(stock_ticker, stock_df, limit=1000):
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
                is_finance_related(clean_body, finance_keywords)
                and len(clean_body.split()) > 5
                and comment.author not in ["AutoModerator", None]
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

    sentiment = df_comments.groupby("date")["Sentiment"].value_counts().unstack()
    for col in ["positive", "neutral", "negative"]:
        if col not in sentiment:
            sentiment[col] = 0
    sentiment = sentiment[["positive", "neutral", "negative"]].div(sentiment.sum(axis=1), axis=0)
    sentiment = sentiment.reindex(stock_df["date"].dt.date, fill_value=0)
    sentiment.reset_index(inplace=True)
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment.reset_index(drop=True, inplace=True)
    return sentiment


# ============ FEATURE MERGING & TRAINING/EVALUATION (new code) ============

def prepare_dataset(stock_df, sentiment):
    merged = pd.merge(stock_df, sentiment, on="date", how="left").fillna(0)
    merged["Lag1"] = merged["Return"].shift(1)
    merged = merged.dropna()
    return merged


def train_and_evaluate(merged):
    X = merged[["Lag1", "positive", "neutral", "negative"]]
    y = merged["Return"]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

    split = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test)

    rmse = mean_squared_error(y_test_inv, y_pred) ** 0.5
    mae = mean_absolute_error(y_test_inv, y_pred)
    r2 = r2_score(y_test_inv, y_pred)

    print("\n📊 Evaluation Metrics:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")

    # DM Test vs baseline lag-1 predictor
    baseline_pred = merged["Return"].shift(1).iloc[-len(y_test_inv):].values.reshape(-1, 1)
    d = (y_test_inv - y_pred.flatten())**2 - (y_test_inv - baseline_pred.flatten())**2
    dm_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    from scipy.stats import t
    p_value = 2 * t.cdf(-abs(dm_stat), df=len(d)-1)

    print("\n⚖️ Diebold–Mariano Test (vs baseline lag-1 predictor):")
    print(f"DM statistic: {dm_stat:.4f}, p-value: {p_value:.4f}")

    # Next-day prediction
    next_features = X_scaled[-1].reshape(1, -1)
    next_pred_scaled = model.predict(next_features)
    next_pred = scaler_y.inverse_transform(next_pred_scaled)
    print("\n🔮 Predicted next-day return:", float(next_pred))

    return model, scaler_X, scaler_y


# ============ RUN PIPELINE ============
if __name__ == "__main__":
    while True:
        ticker = input("\nEnter stock ticker (or 'quit' to exit): ").upper().strip()
        if ticker == "QUIT":
            print("Exiting program.")
            break
        try:
            stock_df = fetch_stock_data(ticker)
            sentiment = fetch_reddit_comments(ticker, stock_df)
            merged = prepare_dataset(stock_df, sentiment)
            model, scaler_X, scaler_y = train_and_evaluate(merged)
        except Exception as e:
            print(f"⚠️ Error processing {stock_ticker}: {e}")
