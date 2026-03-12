import torch
import os
from pathlib import Path


class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data file path (Parquet)
    DATA_DIR = Path(os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))
    DATA_FILE = DATA_DIR / "btc_ohlcv.parquet"

    # Symbol
    SYMBOL = "BTC/USDT:USDT"

    # Training
    BATCH_SIZE = 8192
    TRAIN_STEPS = 1000
    MAX_FORMULA_LEN = 12

    # Feature dimension: RET, LIQ_SCORE, PRESSURE, FOMO, DEV, LOG_VOL, FUNDING_RATE
    INPUT_DIM = 7

    # Backtest
    TRADE_SIZE_USD = 1000.0
    MIN_QUOTE_VOLUME = 1_000_000.0  # Minimum hourly quote volume (USDT) for safe trading
    BASE_FEE = 0.0004               # Binance perp taker fee 0.04%

    # Train / Val / Test split dates (UTC)
    TRAIN_END = "2023-12-31 23:59:59+00"
    VAL_END = "2024-09-30 23:59:59+00"

    # Validation frequency during training
    VAL_EVERY_N_STEPS = 50