import yfinance as yf
import praw
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import pipeline
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
import datetime

# ============ Sentiment Setup ============
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="YOUR_USER_AGENT"
)
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

# ============ LSTM Model ============
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ============ DM Test ============
def diebold_mariano_test(e1, e2, h=1, crit="MSE"):
    d = e1**2 - e2**2 if crit == "MSE" else np.abs(e1) - np.abs(e2)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) / len(d)
    DM_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(DM_stat)))
    return DM_stat, p_value

# ============ Pipeline ============
def fetch_stock_data(ticker, period="1y"):
    df = yf.download(ticker, period=period)
    df["Return"] = df["Close"].pct_change()
    df["Volume_Change"] = df["Volume"].pct_change()
    return df.dropna()

def fetch_reddit_sentiment(ticker, limit=200):
    sentiments = []
    for submission in reddit.subreddit("stocks").search(ticker, limit=limit):
        text = submission.title + " " + (submission.selftext or "")
        result = sentiment_analyzer(text[:512])[0]
        sentiments.append({
            "date": datetime.datetime.utcfromtimestamp(submission.created_utc).date(),
            "sentiment": 1 if result["label"] == "positive" else -1 if result["label"] == "negative" else 0
        })
    return pd.DataFrame(sentiments).groupby("date").mean().reset_index()

def prepare_data(stock_df, sentiment_df, seq_length=10):
    df = stock_df.reset_index()
    df["Date"] = df["Date"].dt.date
    df = pd.merge(df, sentiment_df, left_on="Date", right_on="date", how="left").drop(columns="date")
    df["sentiment"].fillna(0, inplace=True)

    features = ["Return", "Volume_Change", "sentiment"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length, 0])  # return
    return np.array(X), np.array(y), scaler

def train_model(X, y, input_dim, hidden_dim=64, epochs=10, lr=0.001):
    model = LSTMModel(input_dim, hidden_dim, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    return model

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32)).numpy().flatten()
    return preds

# ============ Run ============
if __name__ == "__main__":
    ticker = input("Enter stock ticker (e.g., NVDA, AAPL): ").upper()

    print(f"\nFetching data for {ticker}...")
    stock_df = fetch_stock_data(ticker)
    sentiment_df = fetch_reddit_sentiment(ticker)

    X, y, scaler = prepare_data(stock_df, sentiment_df)
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = train_model(X_train, y_train, input_dim=X.shape[2])

    preds = evaluate(model, X_test, y_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("\n📊 Evaluation Metrics:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")

    # Baseline: last return as prediction
    baseline_preds = np.roll(y_test, 1)
    baseline_preds[0] = y_train[-1]  # set first baseline value
    baseline_errors = y_test - baseline_preds
    model_errors = y_test - preds

    DM_stat, p_val = diebold_mariano_test(model_errors, baseline_errors)

    print("\n⚖️ Diebold–Mariano Test (vs baseline lag-1 predictor):")
    print(f"DM statistic: {DM_stat:.4f}, p-value: {p_val:.4f}")

    # Next-day prediction
    next_input = torch.tensor(X[-1:], dtype=torch.float32)
    next_pred = model(next_input).item()
    print(f"\n🔮 Predicted next-day return for {ticker}: {next_pred:.6f}")
