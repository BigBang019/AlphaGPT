import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Data storage directory
    DATA_DIR = Path(os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
    DATA_FILE = DATA_DIR / "btc_ohlcv.parquet"

    # CCXT Settings
    EXCHANGE = "binance"
    SYMBOL = "BTC/USDT:USDT"     # Perpetual swap pair (linear USDT-margined)
    MARKET_TYPE = "swap"          # 'swap' for perpetual futures
    TIMEFRAME = "1h"
    HISTORY_START = "2020-01-01T00:00:00Z"  # Start date for data download
    CCXT_RATE_LIMIT = True

    # Data split dates (inclusive)
    TRAIN_END = "2023-12-31T23:59:59Z"
    VAL_END = "2024-09-30T23:59:59Z"
    # Everything after VAL_END is test

    # Download batch size (ccxt fetch_ohlcv max per request)
    OHLCV_BATCH_LIMIT = 1500
    FUNDING_BATCH_LIMIT = 1000

    # Concurrency
    CONCURRENCY = 5