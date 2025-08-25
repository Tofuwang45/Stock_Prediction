from __future__ import annotations
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import praw

from config import (
    DATA_DIR, TICKER, WINDOW, EPOCHS, VAL_SPLIT, FINANCE_KEYWORDS,
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT,
    QUERY, TIME_FILTER, LIMIT
)

# Paths
MODEL_PATH = DATA_DIR / "lstm_model.keras"
SCALER_PATH = DATA_DIR / "scaler.pkl"
MERGED_PATH = DATA_DIR / "merged.parquet"
COMMENTS_PATH = DATA_DIR / "comments.parquet"
META_PATH = DATA_DIR / "meta.json"


# ---------- Sentiment (FinBERT) ----------
class FinSentiment:
    """Wrapper for FinBERT sentiment classification."""
    _model = None
    _tokenizer = None
    _labels = ["positive", "neutral", "negative"]

    @classmethod
    def load(cls):
        if cls._model is None or cls._tokenizer is None:
            cls._model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
            cls._tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        return cls

    @classmethod
    def predict_label(cls, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "neutral"
        cls.load()
        inputs = cls._tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = cls._model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return cls._labels[probs.argmax().item()]


# ---------- Reddit ----------
def build_reddit():
    """Build Reddit API client if credentials exist."""
    if not (REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET and REDDIT_USER_AGENT):
        return None
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )


def is_finance_related(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in FINANCE_KEYWORDS)


def fetch_comments() -> pd.DataFrame:
    """Fetch Reddit comments related to ticker."""
    reddit = build_reddit()
    if reddit is None:
        return pd.DataFrame(columns=[
            "post_id", "comment_id", "comment", "score", "author", "created_utc", "sentiment"
        ])

    records = []
    subreddit = reddit.subreddit("all")
    for submission in subreddit.search(QUERY, sort="relevance", time_filter=TIME_FILTER, limit=LIMIT):
        submission.comments.replace_more(limit=0)
        for c in submission.comments.list():
            if getattr(c, "author", None) in ["AutoModerator", None]:
                continue
            body = (c.body or "").replace("\n", " ").strip()
            if len(body.split()) <= 5 or not is_finance_related(body):
                continue
            records.append({
                "post_id": submission.id,
                "comment_id": c.id,
                "comment": body,
                "score": int(c.score),
                "author": str(c.author),
                "created_utc": pd.to_datetime(c.created_utc, unit="s", utc=True)
            })

    df = pd.DataFrame(records)
    if not df.empty:
        df["sentiment"] = df["comment"].apply(FinSentiment.predict_label)
    return df


# ---------- Prices ----------
def fetch_prices() -> pd.DataFrame:
    """Download historical prices for ticker."""
    df = yf.download(TICKER, period="5y", timeout=30)[["Close", "Volume"]].copy()
    df["Return"] = df["Close"].pct_change()
    df = df.reset_index().rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df


# ---------- Sentiment Aggregation ----------
def build_daily_sentiment(df_comments: pd.DataFrame, idx_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """Aggregate daily sentiment and align with price dates."""
    if df_comments.empty:
        return pd.DataFrame({
            "date": [d.date() for d in idx_dates],
            "positive": 0.0, "neutral": 0.0, "negative": 0.0
        })

    tmp = df_comments.copy()
    tmp["date"] = tmp["created_utc"].dt.tz_convert("UTC").dt.date
    ct = tmp.groupby("date")["sentiment"].value_counts().unstack(fill_value=0)

    for lbl in ["positive", "neutral", "negative"]:
        if lbl not in ct.columns:
            ct[lbl] = 0

    denom = ct.sum(axis=1).replace(0, 1)
    ct = ct[["positive", "neutral", "negative"]].div(denom, axis=0)

    idx = pd.Index([d.date() for d in idx_dates], name="date")
    return ct.reindex(idx, fill_value=0).reset_index()


# ---------- Dataset ----------
def make_dataset(merged: pd.DataFrame, features: List[str], target: str, window: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Build dataset with sliding window for LSTM."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(merged[features].values)

    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i, features.index(target)])
    return np.array(X), np.array(y), scaler


def train_lstm(X: np.ndarray, y: np.ndarray, epochs: int, val_split: float):
    """Train LSTM model."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, validation_split=val_split, shuffle=False, verbose=0)
    return model


def attach_last_prediction(merged: pd.DataFrame, model, scaler: MinMaxScaler, features: List[str], window: int) -> Tuple[float, float]:
    """Predict next-day return & close price using latest window."""
    last_window = merged[features].values[-window:]
    scaled_last = scaler.transform(last_window)
    X_last = np.expand_dims(scaled_last, axis=0)

    pred_ret_scaled = float(model.predict(X_last, verbose=0).flatten()[0])

    # Inverse scale prediction
    last_row_scaled = scaler.transform(merged[features].values[-1:].copy())
    last_row_scaled[0, features.index("Return")] = pred_ret_scaled
    inv = scaler.inverse_transform(last_row_scaled)[0]
    predicted_next_return = float(inv[features.index("Return")])

    last_close = float(merged["Close"].iloc[-1])
    predicted_next_close = float(last_close * (1 + predicted_next_return))

    merged.loc[merged.index[-1], "predicted_next_close"] = predicted_next_close
    return predicted_next_return, predicted_next_close


# ---------- State Persistence ----------
def save_meta(meta: Dict[str, Any]):
    META_PATH.write_text(json.dumps(meta, indent=2))


def load_meta() -> Dict[str, Any]:
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


# ---------- Orchestrator ----------
def run_update() -> Dict[str, Any]:
    """Run full pipeline: fetch → train → predict → persist."""
    start = time.time()

    prices = fetch_prices()
    comments = fetch_comments()
    comments.to_parquet(COMMENTS_PATH, index=False)

    sent_daily = build_daily_sentiment(comments, prices["date"].tolist())
    merged = prices.merge(sent_daily, left_on=prices["date"].dt.date, right_on="date", how="left")
    merged.rename(columns={"date_x": "date"}, inplace=True)
    merged = merged.drop(columns=["key_0"], errors="ignore")
    merged[["positive", "neutral", "negative"]] = merged[["positive", "neutral", "negative"]].fillna(0.0)

    # dataset
    features = ["Close", "Volume", "Return", "positive", "neutral", "negative"]
    merged = merged.dropna(subset=["Return"]).reset_index(drop=True)
    X, y, scaler = make_dataset(merged, features, "Return", WINDOW)

    model = train_lstm(X, y, EPOCHS, VAL_SPLIT)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    pred_ret, pred_close = attach_last_prediction(merged, model, scaler, features, WINDOW)
    merged.to_parquet(MERGED_PATH, index=False)

    meta = {
        "ticker": TICKER,
        "last_update_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(merged),
        "window": WINDOW,
        "epochs": EPOCHS,
        "predicted_next_return": pred_ret,
        "predicted_next_close": pred_close,
        "comments": len(comments),
        "elapsed_sec": round(time.time() - start, 2),
    }
    save_meta(meta)
    return meta


def load_state() -> Tuple[pd.DataFrame, Dict[str, Any], Any, MinMaxScaler]:
    merged = pd.read_parquet(MERGED_PATH) if MERGED_PATH.exists() else pd.DataFrame()
    meta = load_meta()
    model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return merged, meta, model, scaler


def recent_comments(limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent high-score comments."""
    if not COMMENTS_PATH.exists():
        return []
    df = pd.read_parquet(COMMENTS_PATH)
    if df.empty:
        return []

    df = df.sort_values("score", ascending=False).head(limit)
    return [
        {
            "post_id": r.post_id,
            "comment_id": r.comment_id,
            "author": r.author,
            "score": int(r.score),
            "sentiment": r.sentiment,
            "created_utc": pd.to_datetime(r.created_utc).isoformat(),
        }
        for _, r in df.iterrows()
    ]
