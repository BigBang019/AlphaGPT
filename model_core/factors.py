import torch
import torch.nn as nn


class RMSNormFactor(nn.Module):
    """RMSNorm for factor normalization"""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MemeIndicators:
    @staticmethod
    def liquidity_health(quote_volume, fdv):
        """Liquidity health score based on quote volume vs FDV ratio."""
        ratio = quote_volume / (fdv + 1e-6)
        return torch.clamp(ratio * 4.0, 0.0, 1.0)

    @staticmethod
    def buy_sell_imbalance(close, open_, high, low):
        range_hl = high - low + 1e-9
        body = close - open_
        strength = body / range_hl
        return torch.tanh(strength * 3.0)

    @staticmethod
    def fomo_acceleration(volume, window=5):
        vol_prev = torch.roll(volume, 1, dims=1)
        vol_chg = (volume - vol_prev) / (vol_prev + 1.0)
        acc = vol_chg - torch.roll(vol_chg, 1, dims=1)
        return torch.clamp(acc, -5.0, 5.0)

    @staticmethod
    def pump_deviation(close, window=20):
        pad = torch.zeros((close.shape[0], window - 1), device=close.device)
        c_pad = torch.cat([pad, close], dim=1)
        ma = c_pad.unfold(1, window, 1).mean(dim=-1)
        dev = (close - ma) / (ma + 1e-9)
        return dev

    @staticmethod
    def volatility_clustering(close, window=10):
        """Detect volatility clustering patterns"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))
        ret_sq = ret**2

        pad = torch.zeros((ret_sq.shape[0], window - 1), device=close.device)
        ret_sq_pad = torch.cat([pad, ret_sq], dim=1)
        vol_ma = ret_sq_pad.unfold(1, window, 1).mean(dim=-1)

        return torch.sqrt(vol_ma + 1e-9)

    @staticmethod
    def momentum_reversal(close, window=5):
        """Capture momentum reversal signals"""
        ret = torch.log(close / (torch.roll(close, 1, dims=1) + 1e-9))

        pad = torch.zeros((ret.shape[0], window - 1), device=close.device)
        ret_pad = torch.cat([pad, ret], dim=1)
        mom = ret_pad.unfold(1, window, 1).sum(dim=-1)

        # Detect reversals
        mom_prev = torch.roll(mom, 1, dims=1)
        reversal = (mom * mom_prev < 0).float()

        return reversal

    @staticmethod
    def relative_strength(close, high, low, window=14):
        """RSI-like indicator for strength detection"""
        ret = close - torch.roll(close, 1, dims=1)

        gains = torch.relu(ret)
        losses = torch.relu(-ret)

        pad = torch.zeros((gains.shape[0], window - 1), device=close.device)
        gains_pad = torch.cat([pad, gains], dim=1)
        losses_pad = torch.cat([pad, losses], dim=1)

        avg_gain = gains_pad.unfold(1, window, 1).mean(dim=-1)
        avg_loss = losses_pad.unfold(1, window, 1).mean(dim=-1)

        rs = (avg_gain + 1e-9) / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return (rsi - 50) / 50  # Normalize


class FeatureEngineer:
    """
    Feature engineering for BTC/USDT perpetual.
    Produces 7-dimensional features:
      0: RET        - log return (robust normalized)
      1: LIQ_SCORE  - quote_volume / fdv liquidity health
      2: PRESSURE   - buy/sell pressure from candle body
      3: FOMO       - volume acceleration
      4: DEV        - price deviation from MA
      5: LOG_VOL    - log volume (robust normalized)
      6: FR         - funding rate (robust normalized)
    """

    INPUT_DIM = 7

    @staticmethod
    def compute_features(raw_dict):
        c = raw_dict["close"]
        o = raw_dict["open"]
        h = raw_dict["high"]
        l = raw_dict["low"]
        v = raw_dict["volume"]
        qv = raw_dict["quote_volume"]
        fdv = raw_dict["fdv"]
        fr = raw_dict["funding_rate"]

        # Log returns
        ret = torch.log(c / (torch.roll(c, 1, dims=1) + 1e-9))

        # Liquidity health (quote_volume vs fdv)
        liq_score = MemeIndicators.liquidity_health(qv, fdv)

        # Buy/sell pressure
        pressure = MemeIndicators.buy_sell_imbalance(c, o, h, l)

        # FOMO acceleration
        fomo = MemeIndicators.fomo_acceleration(v)

        # Price deviation from MA
        dev = MemeIndicators.pump_deviation(c)

        # Log volume
        log_vol = torch.log1p(v)

        # Funding rate (already a clean scalar per timestep)
        funding = fr

        def robust_norm(t):
            median = torch.nanmedian(t, dim=1, keepdim=True)[0]
            mad = torch.nanmedian(torch.abs(t - median), dim=1, keepdim=True)[0] + 1e-6
            norm = (t - median) / mad
            return torch.clamp(norm, -5.0, 5.0)

        features = torch.stack(
            [
                robust_norm(ret),       # 0: RET
                liq_score,              # 1: LIQ_SCORE
                pressure,               # 2: PRESSURE
                robust_norm(fomo),      # 3: FOMO
                robust_norm(dev),       # 4: DEV
                robust_norm(log_vol),   # 5: LOG_VOL
                robust_norm(funding),   # 6: FR (funding rate)
            ],
            dim=1,
        )

        return features