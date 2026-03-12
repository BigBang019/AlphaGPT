import torch
from .config import ModelConfig


class BTCBacktest:
    """
    Backtester for BTC/USDT perpetual futures alpha factors.

    Uses quote_volume instead of DEX liquidity, and Binance perp fee structure.
    """

    def __init__(self):
        self.trade_size = ModelConfig.TRADE_SIZE_USD
        self.min_quote_vol = ModelConfig.MIN_QUOTE_VOLUME
        self.base_fee = ModelConfig.BASE_FEE

    def evaluate(self, factors, raw_data, target_ret):
        """
        Evaluate a factor signal on historical data.

        Args:
            factors: [N, T] tensor of factor values
            raw_data: dict with 'quote_volume' etc, each [N, T]
            target_ret: [N, T] tensor of forward returns

        Returns:
            (score, cum_return_mean) tuple
        """
        quote_vol = raw_data["quote_volume"]

        # Signal: sigmoid of factor values -> [0, 1]
        signal = torch.sigmoid(factors)

        # Safety: only trade when quote volume is sufficient
        is_safe = (quote_vol > self.min_quote_vol).float()

        # Position: go long when signal > 0.6, safe to trade
        # (lowered threshold from 0.85 since BTC is more liquid and less memey)
        position = (signal > 0.6).float() * is_safe

        # Market impact: negligible for BTC, but model it anyway
        impact_slippage = self.trade_size / (quote_vol + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.01)

        # Total one-way cost
        total_slippage_one_way = self.base_fee + impact_slippage

        # Turnover
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way

        # PnL
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost

        # Cumulative return
        cum_ret = net_pnl.sum(dim=1)

        # Penalize large drawdowns
        big_drawdowns = (net_pnl < -0.02).float().sum(dim=1)  # Tighter threshold for BTC
        score = cum_ret - (big_drawdowns * 1.0)

        # Penalize inactivity
        activity = position.sum(dim=1)
        score = torch.where(
            activity < 5, torch.tensor(-10.0, device=score.device), score
        )

        final_fitness = torch.median(score)
        return final_fitness, cum_ret.mean().item()