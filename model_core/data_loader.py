import pandas as pd
import torch
from loguru import logger
from .config import ModelConfig
from .factors import FeatureEngineer


class CryptoDataLoader:
    """
    Data loader for BTC/USDT perpetual with train/val/test split.

    Loads OHLCV + funding_rate from Parquet file, computes features,
    and splits along the time dimension.
    """

    def __init__(self):
        # Full data
        self.feat_tensor = None
        self.raw_data_cache = None
        self.target_ret = None

        # Split data
        self.train_feat = None
        self.train_raw = None
        self.train_target = None

        self.val_feat = None
        self.val_raw = None
        self.val_target = None

        self.test_feat = None
        self.test_raw = None
        self.test_target = None

        # Split indices
        self.train_end_idx = None
        self.val_end_idx = None

    def load_data(self):
        """Load data from Parquet file and compute features + splits."""
        data_path = ModelConfig.DATA_FILE
        logger.info(f"Loading BTC/USDT data from {data_path}...")

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                f"Please run the data pipeline first: python -m data_pipeline.run_pipeline"
            )

        df = pd.read_parquet(data_path)

        # Filter by symbol (in case file contains multiple)
        if "symbol" in df.columns:
            df = df[df["symbol"] == ModelConfig.SYMBOL].copy()

        if df.empty:
            raise ValueError(f"No data found for {ModelConfig.SYMBOL} in {data_path}")

        # Ensure sorted by time
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        # Convert to tensors — for single symbol, shape is [1, T]
        device = ModelConfig.DEVICE

        def to_tensor(col):
            return torch.tensor(
                df[col].values.reshape(1, -1), dtype=torch.float32, device=device
            )

        self.raw_data_cache = {
            "open": to_tensor("open"),
            "high": to_tensor("high"),
            "low": to_tensor("low"),
            "close": to_tensor("close"),
            "volume": to_tensor("volume"),
            "quote_volume": to_tensor("quote_volume"),
            "fdv": to_tensor("fdv"),
            "funding_rate": to_tensor("funding_rate"),
        }

        # Compute features: [1, 7, T]
        self.feat_tensor = FeatureEngineer.compute_features(self.raw_data_cache)

        # Compute target return: log(open[t+2] / open[t+1])
        op = self.raw_data_cache["open"]
        t1 = torch.roll(op, -1, dims=1)
        t2 = torch.roll(op, -2, dims=1)
        self.target_ret = torch.log(t2 / (t1 + 1e-9))
        self.target_ret[:, -2:] = 0.0

        # --- Time-based split ---
        train_end_dt = pd.Timestamp(ModelConfig.TRAIN_END)
        val_end_dt = pd.Timestamp(ModelConfig.VAL_END)

        times = df["timestamp"]
        self.train_end_idx = int((times <= train_end_dt).sum())
        self.val_end_idx = int((times <= val_end_dt).sum())
        total_T = len(df)

        logger.info(
            f"Split: Train [0:{self.train_end_idx}] ({self.train_end_idx} bars) | "
            f"Val [{self.train_end_idx}:{self.val_end_idx}] ({self.val_end_idx - self.train_end_idx} bars) | "
            f"Test [{self.val_end_idx}:{total_T}] ({total_T - self.val_end_idx} bars)"
        )

        # Split features [1, F, T] → slice on T dimension
        self.train_feat = self.feat_tensor[:, :, : self.train_end_idx]
        self.val_feat = self.feat_tensor[:, :, self.train_end_idx : self.val_end_idx]
        self.test_feat = self.feat_tensor[:, :, self.val_end_idx :]

        # Split target return [1, T]
        self.train_target = self.target_ret[:, : self.train_end_idx]
        self.val_target = self.target_ret[:, self.train_end_idx : self.val_end_idx]
        self.test_target = self.target_ret[:, self.val_end_idx :]

        # Split raw data cache
        self.train_raw = self._slice_raw(0, self.train_end_idx)
        self.val_raw = self._slice_raw(self.train_end_idx, self.val_end_idx)
        self.test_raw = self._slice_raw(self.val_end_idx, total_T)

        logger.success(
            f"Data ready. Feature shape: {self.feat_tensor.shape} "
            f"(Train: {self.train_feat.shape[2]}, Val: {self.val_feat.shape[2]}, Test: {self.test_feat.shape[2]})"
        )

    def _slice_raw(self, start: int, end: int) -> dict:
        """Slice raw_data_cache along the time dimension."""
        return {k: v[:, start:end] for k, v in self.raw_data_cache.items()}