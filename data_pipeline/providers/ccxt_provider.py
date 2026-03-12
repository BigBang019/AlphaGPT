import ccxt
import time
import pandas as pd
from datetime import datetime, timezone
from loguru import logger
from ..config import Config


class CCXTProvider:
    """
    Data provider using ccxt to download BTC/USDT perpetual data from Binance.
    Downloads OHLCV candles and funding rate history.
    """

    def __init__(self):
        exchange_class = getattr(ccxt, Config.EXCHANGE)
        self.exchange = exchange_class({
            "enableRateLimit": Config.CCXT_RATE_LIMIT,
            "options": {
                "defaultType": Config.MARKET_TYPE,  # 'swap' for perpetual
            },
        })
        self.symbol = Config.SYMBOL
        self.timeframe = Config.TIMEFRAME

    def _parse_start_ms(self) -> int:
        """Parse HISTORY_START config to millisecond timestamp."""
        dt = datetime.fromisoformat(Config.HISTORY_START.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)

    def fetch_all_ohlcv(self) -> pd.DataFrame:
        """
        Fetch all OHLCV data from HISTORY_START to now.
        Handles pagination automatically.

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Fetching OHLCV for {self.symbol} ({self.timeframe}) from {Config.HISTORY_START}...")

        since = self._parse_start_ms()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_candles = []

        while since < now_ms:
            try:
                candles = self.exchange.fetch_ohlcv(
                    self.symbol,
                    timeframe=self.timeframe,
                    since=since,
                    limit=Config.OHLCV_BATCH_LIMIT,
                )
                if not candles:
                    break

                all_candles.extend(candles)
                # Move since to after the last candle timestamp
                since = candles[-1][0] + 1
                logger.debug(f"  Fetched {len(candles)} candles, total: {len(all_candles)}, last: {datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc).isoformat()}")

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limited, sleeping 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"OHLCV fetch error: {e}")
                time.sleep(5)
                # Move forward to avoid infinite loop
                since += 3600 * 1000 * Config.OHLCV_BATCH_LIMIT

        if not all_candles:
            logger.error("No OHLCV data fetched!")
            return pd.DataFrame()

        df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # Compute quote_volume (proxy for liquidity) and fdv
        df["quote_volume"] = df["close"] * df["volume"]
        df["fdv"] = df["close"] * 21_000_000.0  # BTC max supply

        logger.success(f"OHLCV complete: {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df

    def fetch_all_funding_rates(self) -> pd.DataFrame:
        """
        Fetch funding rate history from HISTORY_START to now.
        Binance perpetual funding rates are typically every 8 hours.

        Returns:
            DataFrame with columns: [timestamp, funding_rate]
        """
        logger.info(f"Fetching funding rates for {self.symbol} from {Config.HISTORY_START}...")

        since = self._parse_start_ms()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        all_rates = []

        while since < now_ms:
            try:
                # ccxt unified method for funding rate history
                rates = self.exchange.fetch_funding_rate_history(
                    self.symbol,
                    since=since,
                    limit=Config.FUNDING_BATCH_LIMIT,
                )
                if not rates:
                    break

                for r in rates:
                    all_rates.append({
                        "timestamp": r["timestamp"],
                        "funding_rate": r.get("fundingRate", 0.0),
                    })

                since = rates[-1]["timestamp"] + 1
                logger.debug(f"  Fetched {len(rates)} funding records, total: {len(all_rates)}")

            except ccxt.RateLimitExceeded:
                logger.warning("Rate limited on funding, sleeping 10s...")
                time.sleep(10)
            except Exception as e:
                logger.error(f"Funding rate fetch error: {e}")
                time.sleep(5)
                # Funding rates are every 8h, skip ~1000 periods
                since += 8 * 3600 * 1000 * Config.FUNDING_BATCH_LIMIT

        if not all_rates:
            logger.warning("No funding rate data fetched.")
            return pd.DataFrame()

        df = pd.DataFrame(all_rates)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        logger.success(f"Funding rates complete: {len(df)} records from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df