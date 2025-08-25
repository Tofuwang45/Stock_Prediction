import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

load_dotenv(dotenv_path=BASE_DIR / ".env")

# Reddit (optional - if unset, app runs w/o comments)
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "nvda-sentiment-app/1.0 by u/youruser")

# App
TICKER = os.getenv("TICKER", "NVDA")
QUERY = os.getenv("REDDIT_QUERY", '(title:"NVDA" OR selftext:"NVDA") AND (title:stock OR title:share OR title:earnings OR title:market)')
TIME_FILTER = os.getenv("REDDIT_TIME_FILTER", "month")   # day, week, month, year, all
LIMIT = int(os.getenv("REDDIT_LIMIT", "500"))            # posts to scan
WINDOW = int(os.getenv("LSTM_WINDOW", "30"))
EPOCHS = int(os.getenv("LSTM_EPOCHS", "8"))
VAL_SPLIT = float(os.getenv("LSTM_VAL_SPLIT", "0.2"))

FINANCE_KEYWORDS = [w.strip() for w in os.getenv("FINANCE_KEYWORDS",
    "stock,share,ticker,dividend,price,earnings,revenue,guidance,options,puts,calls,market,invest,investing,hold,buy,sell,portfolio"
).split(",") if w.strip()]
