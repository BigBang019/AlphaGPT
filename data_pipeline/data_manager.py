import pandas as pd
from pathlib import Path
from loguru import logger
from .config import Config
from .providers.ccxt_provider import CCXTProvider


class DataManager:
    """
    Data manager that downloads BTC/USDT perp data via ccxt
    and stores it as a Parquet file. Fully synchronous — no database needed.
    """

    def __init__(self):
        self.provider = CCXTProvider()

    def pipeline_sync(self):
        """
        Main pipeline: download BTC/USDT perp OHLCV + funding rates from Binance,
        merge them, and save to Parquet file.
        """
        # --- Step 1: Fetch OHLCV ---
        logger.info("Step 1: Fetching OHLCV data via ccxt...")
        ohlcv_df = self.provider.fetch_all_ohlcv()
        if ohlcv_df.empty:
            logger.error("No OHLCV data. Aborting.")
            return

        # --- Step 2: Fetch Funding Rates ---
        logger.info("Step 2: Fetching funding rate data via ccxt...")
        funding_df = self.provider.fetch_all_funding_rates()

        # --- Step 3: Merge funding rates into OHLCV ---
        logger.info("Step 3: Merging funding rates into OHLCV...")
        if not funding_df.empty:
            # Funding rates come every 8h; forward-fill to align with 1h OHLCV
            funding_df = funding_df.set_index("timestamp").sort_index()
            ohlcv_df = ohlcv_df.set_index("timestamp").sort_index()

            # Reindex funding to OHLCV timestamps with forward fill
            ohlcv_df["funding_rate"] = (
                funding_df["funding_rate"]
                .reindex(ohlcv_df.index, method="ffill")
                .fillna(0.0)
                .values
            )
            ohlcv_df = ohlcv_df.reset_index()
        else:
            ohlcv_df["funding_rate"] = 0.0

        # Add symbol column
        ohlcv_df["symbol"] = Config.SYMBOL

        # --- Step 4: Save to Parquet ---
        data_dir = Path(Config.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)

        output_path = Config.DATA_FILE
        ohlcv_df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.success(
            f"Pipeline complete. Saved {len(ohlcv_df)} records to {output_path}\n"
            f"  Range: {ohlcv_df['timestamp'].iloc[0]} to {ohlcv_df['timestamp'].iloc[-1]}\n"
            f"  Columns: {list(ohlcv_df.columns)}\n"
            f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB"
        )