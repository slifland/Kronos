from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kronos_trading.backtest import (
    BacktestConfig,
    SingleAssetBacktester,
    compute_equal_weight_benchmark,
)
from kronos_trading.signals import KronosForecastStrategy, KronosStrategyConfig


class DummyPredictor:
    def __init__(self, forecast_close: float) -> None:
        self.forecast_close = forecast_close

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose):
        return pd.DataFrame(
            {
                "open": [self.forecast_close] * pred_len,
                "high": [self.forecast_close] * pred_len,
                "low": [self.forecast_close] * pred_len,
                "close": [self.forecast_close] * pred_len,
                "volume": [0.0] * pred_len,
                "amount": [0.0] * pred_len,
            },
            index=y_timestamp,
        )


class AlwaysLongStrategy:
    def __init__(self, lookback: int) -> None:
        self.config = type("Config", (), {"lookback": lookback, "pred_len": 1})()

    def generate_signal(self, history, future_timestamps):
        return type(
            "Decision",
            (),
            {
                "target_exposure": 1.0,
                "predicted_return": 0.01,
                "forecast_close": float(history["close"].iloc[-1]) * 1.01,
                "reference_close": float(history["close"].iloc[-1]),
            },
        )()


def test_kronos_forecast_strategy_long_signal():
    history_index = pd.date_range("2025-01-02 09:30", periods=3, freq="5min", tz="America/New_York")
    history = pd.DataFrame(
        {
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1000, 1000],
        },
        index=history_index,
    )
    future_index = pd.date_range("2025-01-02 09:45", periods=1, freq="5min", tz="America/New_York")

    strategy = KronosForecastStrategy(
        predictor=DummyPredictor(forecast_close=103.0),
        config=KronosStrategyConfig(lookback=3, pred_len=1, signal_threshold=0.005),
    )

    decision = strategy.generate_signal(history=history, future_timestamps=future_index)

    assert decision.target_exposure == 1.0
    assert decision.predicted_return > 0.005


def test_single_asset_backtester_tracks_long_only_equity():
    index = pd.date_range("2025-01-02 09:30", periods=4, freq="5min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "open": [100.0, 110.0, 121.0, 133.1],
            "high": [100.0, 110.0, 121.0, 133.1],
            "low": [100.0, 110.0, 121.0, 133.1],
            "close": [100.0, 110.0, 121.0, 133.1],
            "volume": [1000, 1000, 1000, 1000],
        },
        index=index,
    )

    result = SingleAssetBacktester(
        strategy=AlwaysLongStrategy(lookback=2),
        config=BacktestConfig(initial_capital=100000.0, fee_bps=0.0, slippage_bps=0.0, periods_per_year=252.0),
    ).run(bars, symbol="AAPL")

    assert round(result.results["equity"].iloc[-1], 2) == 121000.0
    assert result.metrics["trade_count"] == 1.0


def test_equal_weight_benchmark_averages_series_returns():
    index = pd.date_range("2025-01-02", periods=3, freq="D")
    close_frame = pd.DataFrame(
        {
            "AAA": [100.0, 110.0, 121.0],
            "BBB": [100.0, 105.0, 110.25],
        },
        index=index,
    )

    curve, metrics = compute_equal_weight_benchmark(
        close_frame=close_frame,
        target_index=index,
        initial_capital=100000.0,
        periods_per_year=252.0,
    )

    assert round(curve["equity"].iloc[-1], 2) == 115562.5
    assert metrics["total_return"] > 0.15
