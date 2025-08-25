from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timezone, time as dtime
import pandas as pd

from pipeline import run_update, load_state, recent_comments
from config import TICKER

app = FastAPI(title="NVDA Sentiment + Price Predictor API", version="1.0.0")

# CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

# Load state on boot
merged, meta, model, scaler = load_state()

def schedule_daily_refresh():
    # run once at startup
    def _refresh():
        global merged, meta, model, scaler
        meta = run_update()
        merged, meta, model, scaler = load_state()

    scheduler = BackgroundScheduler(daemon=True)
    # run at ~ 13:00 UTC daily (before US market open Pacific)
    scheduler.add_job(_refresh, "cron", hour=13, minute=5)
    scheduler.start()
    # also warm up immediately on boot if empty
    if merged is None or getattr(merged, "empty", True):
        _refresh()

schedule_daily_refresh()

@app.get("/health")
def health():
    return {"ok": True, "ticker": TICKER, "last_update": meta.get("last_update_utc") if meta else None}

@app.get("/api/metrics")
def api_metrics():
    if merged is None or merged.empty:
        return {"merged": [], "stats": {}}
    # Serialize a slim version to the UI
    out = []
    for _, r in merged.iterrows():
        out.append({
            "date": pd.to_datetime(r["date"]).date().isoformat(),
            "Close": float(r["Close"]),
            "Volume": int(r["Volume"]),
            "Return": float(r["Return"]),
            "positive": float(r["positive"]),
            "neutral": float(r["neutral"]),
            "negative": float(r["negative"]),
            "predicted_next_close": float(r["predicted_next_close"]) if "predicted_next_close" in merged.columns and pd.notna(r["predicted_next_close"]) else None,
        })

    # Stats
    last30 = merged.tail(30)
    stats = {
        "rows": int(len(merged)),
        "comment_count_30d": int((last30[["positive","neutral","negative"]].sum(axis=1) > 0).sum()),
        "last_date": pd.to_datetime(merged["date"].iloc[-1]).date().isoformat(),
    }
    return {"merged": out, "stats": stats}

@app.get("/api/predict")
def api_predict():
    if not meta:
        return {"predicted_next_return": None, "predicted_next_close": None}
    return {
        "predicted_next_return": meta.get("predicted_next_return"),
        "predicted_next_close": meta.get("predicted_next_close"),
        "last_update_utc": meta.get("last_update_utc"),
    }

@app.get("/api/comments")
def api_comments(limit: int = Query(50, ge=1, le=200)):
    return {"items": recent_comments(limit)}
