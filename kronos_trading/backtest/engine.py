from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from .benchmarks import compute_performance_metrics


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100000.0
    fee_bps: float = 0.0
    slippage_bps: float = 1.0
    periods_per_year: float = 252.0
    show_progress: bool = True


@dataclass
class BacktestResult:
    results: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]


class SingleAssetBacktester:
    def __init__(self, strategy: Any, config: BacktestConfig | None = None) -> None:
        self.strategy = strategy
        self.config = config or BacktestConfig()

    def run(self, bars: pd.DataFrame, symbol: str | None = None) -> BacktestResult:
        frame = bars.copy().sort_index()
        required_columns = {"open", "high", "low", "close"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Backtest bars are missing required columns: {sorted(missing_columns)}")

        lookback = getattr(self.strategy.config, "lookback", None)
        pred_len = getattr(self.strategy.config, "pred_len", 1)
        if lookback is None:
            raise ValueError("Strategy config must define a lookback value.")

        if len(frame) < lookback + pred_len + 1:
            raise ValueError(
                "Not enough bars to backtest. Provide at least lookback + pred_len + 1 rows."
            )

        equity = float(self.config.initial_capital)
        current_exposure = 0.0
        records: list[dict[str, object]] = []
        trades: list[dict[str, object]] = []

        step_iter = range(lookback - 1, len(frame) - pred_len)
        if self.config.show_progress:
            step_iter = tqdm(
                step_iter,
                total=max(len(frame) - pred_len - (lookback - 1), 0),
                desc=f"Backtesting {symbol or 'strategy'}",
                leave=False,
            )

        for index in step_iter:
            history = frame.iloc[index - lookback + 1 : index + 1]
            future_timestamps = frame.index[index + 1 : index + 1 + pred_len]
            decision = self.strategy.generate_signal(history=history, future_timestamps=future_timestamps)

            next_close = float(frame["close"].iloc[index + 1])
            current_close = float(frame["close"].iloc[index])
            asset_return = (next_close / current_close) - 1.0

            turnover = abs(decision.target_exposure - current_exposure)
            cost_fraction = turnover * (self.config.fee_bps + self.config.slippage_bps) / 10000.0
            cost_paid = equity * cost_fraction
            equity_after_cost = equity - cost_paid
            strategy_return = decision.target_exposure * asset_return
            next_equity = equity_after_cost * (1.0 + strategy_return)

            if turnover > 0.0:
                trades.append(
                    {
                        "symbol": symbol or "",
                        "decision_time": frame.index[index],
                        "effective_time": frame.index[index + 1],
                        "from_exposure": current_exposure,
                        "to_exposure": decision.target_exposure,
                        "predicted_return": decision.predicted_return,
                        "forecast_close": decision.forecast_close,
                        "reference_close": decision.reference_close,
                        "cost_paid": cost_paid,
                    }
                )

            records.append(
                {
                    "symbol": symbol or "",
                    "decision_time": frame.index[index],
                    "forecast_time": frame.index[index + 1],
                    "close": current_close,
                    "next_close": next_close,
                    "asset_return": asset_return,
                    "target_exposure": decision.target_exposure,
                    "predicted_return": decision.predicted_return,
                    "forecast_close": decision.forecast_close,
                    "reference_close": decision.reference_close,
                    "turnover": turnover,
                    "cost_paid": cost_paid,
                    "strategy_return": strategy_return,
                    "equity": next_equity,
                }
            )

            current_exposure = decision.target_exposure
            equity = next_equity

            if self.config.show_progress and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix(equity=f"{equity:,.0f}")

        result_frame = pd.DataFrame.from_records(records).set_index("forecast_time")
        trade_frame = pd.DataFrame.from_records(trades)
        metrics = compute_performance_metrics(
            result_frame["equity"],
            result_frame["strategy_return"],
            periods_per_year=self.config.periods_per_year,
            turnover=result_frame["turnover"],
            trade_count=len(trade_frame),
        )

        return BacktestResult(results=result_frame, trades=trade_frame, metrics=metrics)
