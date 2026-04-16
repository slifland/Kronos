from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

TIMEFRAME_PATTERN = re.compile(r"^(?P<count>\d+)(?P<unit>Min|Hour|Day|Week|Month)$", re.IGNORECASE)


@dataclass
class BenchmarkBundle:
    curves: pd.DataFrame
    metrics: dict[str, dict[str, float]]


def infer_periods_per_year(timeframe: str) -> float:
    match = TIMEFRAME_PATTERN.match(timeframe.strip())
    if not match:
        raise ValueError(
            "Unsupported timeframe format. Use Alpaca-style values such as 5Min, 15Min, 1Hour, or 1Day."
        )

    count = int(match.group("count"))
    unit = match.group("unit").lower()

    if unit == "min":
        return 252.0 * (390.0 / count)
    if unit == "hour":
        return 252.0 * (6.5 / count)
    if unit == "day":
        return 252.0 / count
    if unit == "week":
        return 52.0 / count
    if unit == "month":
        return 12.0 / count

    raise ValueError(f"Unsupported timeframe unit: {unit}")


def _build_equity_curve(returns: pd.Series, initial_capital: float) -> pd.DataFrame:
    clean_returns = returns.fillna(0.0).astype(float)
    equity = initial_capital * (1.0 + clean_returns).cumprod()
    return pd.DataFrame({"return": clean_returns, "equity": equity}, index=clean_returns.index)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    drawdown = (equity / equity.cummax()) - 1.0
    return float(drawdown.min())


def compute_performance_metrics(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: float,
    turnover: pd.Series | None = None,
    trade_count: int | None = None,
) -> dict[str, float]:
    clean_returns = returns.fillna(0.0).astype(float)
    if clean_returns.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover_total": 0.0,
            "trade_count": float(trade_count or 0),
        }

    total_return = float((1.0 + clean_returns).prod() - 1.0)
    annualized_return = float((1.0 + total_return) ** (periods_per_year / max(len(clean_returns), 1)) - 1.0)
    annualized_volatility = float(clean_returns.std(ddof=0) * math.sqrt(periods_per_year))

    if annualized_volatility == 0.0:
        sharpe = 0.0
    else:
        sharpe = float((clean_returns.mean() / clean_returns.std(ddof=0)) * math.sqrt(periods_per_year))

    active_returns = clean_returns[clean_returns != 0.0]
    win_rate = float((active_returns > 0.0).mean()) if not active_returns.empty else 0.0
    turnover_total = float(turnover.fillna(0.0).sum()) if turnover is not None else 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "win_rate": win_rate,
        "turnover_total": turnover_total,
        "trade_count": float(trade_count or 0),
    }


def compute_buy_and_hold_benchmark(
    close: pd.Series,
    target_index: pd.Index,
    initial_capital: float,
    periods_per_year: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    aligned_close = close.reindex(target_index).ffill().dropna()
    returns = aligned_close.pct_change().fillna(0.0)
    curve = _build_equity_curve(returns, initial_capital)
    metrics = compute_performance_metrics(curve["equity"], curve["return"], periods_per_year)
    return curve, metrics


def compute_equal_weight_benchmark(
    close_frame: pd.DataFrame,
    target_index: pd.Index,
    initial_capital: float,
    periods_per_year: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    aligned_close = close_frame.reindex(target_index).ffill().dropna(how="all")
    returns = aligned_close.pct_change().fillna(0.0)
    equal_weight_returns = returns.mean(axis=1).fillna(0.0)
    curve = _build_equity_curve(equal_weight_returns, initial_capital)
    metrics = compute_performance_metrics(curve["equity"], curve["return"], periods_per_year)
    return curve, metrics


def compute_momentum_benchmark(
    close: pd.Series,
    target_index: pd.Index,
    initial_capital: float,
    periods_per_year: float,
    lookback_bars: int = 12,
    allow_short: bool = False,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    aligned_close = close.reindex(target_index).ffill().dropna()
    momentum = aligned_close.pct_change(lookback_bars)
    exposure = pd.Series(0.0, index=aligned_close.index)
    exposure[momentum > threshold] = 1.0
    if allow_short:
        exposure[momentum < -threshold] = -1.0

    bar_returns = aligned_close.pct_change().fillna(0.0)
    strategy_returns = exposure.shift(1).fillna(0.0) * bar_returns
    turnover = exposure.diff().abs().fillna(exposure.abs())

    curve = _build_equity_curve(strategy_returns, initial_capital)
    curve["exposure"] = exposure
    curve["turnover"] = turnover
    metrics = compute_performance_metrics(
        curve["equity"],
        curve["return"],
        periods_per_year,
        turnover=turnover,
        trade_count=int((turnover > 0.0).sum()),
    )
    return curve, metrics


def evaluate_benchmarks(
    strategy_index: pd.Index,
    strategy_close: pd.Series,
    benchmark_closes: dict[str, pd.Series],
    initial_capital: float,
    periods_per_year: float,
    momentum_lookback: int = 12,
    allow_short: bool = False,
) -> BenchmarkBundle:
    curves: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, float]] = {}

    buy_hold_curve, buy_hold_metrics = compute_buy_and_hold_benchmark(
        strategy_close,
        target_index=strategy_index,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
    )
    curves["buy_hold"] = buy_hold_curve["equity"]
    metrics["buy_hold"] = buy_hold_metrics

    momentum_curve, momentum_metrics = compute_momentum_benchmark(
        strategy_close,
        target_index=strategy_index,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year,
        lookback_bars=momentum_lookback,
        allow_short=allow_short,
    )
    curves["momentum"] = momentum_curve["equity"]
    metrics["momentum"] = momentum_metrics

    if benchmark_closes:
        close_frame = pd.DataFrame(
            {
                symbol: close.reindex(strategy_index).ffill()
                for symbol, close in benchmark_closes.items()
            }
        )
        for symbol, close in benchmark_closes.items():
            curve, summary = compute_buy_and_hold_benchmark(
                close,
                target_index=strategy_index,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
            )
            curves[symbol] = curve["equity"]
            metrics[symbol] = summary

        if len(close_frame.columns) > 1:
            equal_weight_curve, equal_weight_metrics = compute_equal_weight_benchmark(
                close_frame,
                target_index=strategy_index,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
            )
            curves["equal_weight"] = equal_weight_curve["equity"]
            metrics["equal_weight"] = equal_weight_metrics

    curve_frame = pd.DataFrame(curves).sort_index()
    return BenchmarkBundle(curves=curve_frame, metrics=metrics)
