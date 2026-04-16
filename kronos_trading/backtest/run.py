from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from kronos_trading.backtest.benchmarks import evaluate_benchmarks, infer_periods_per_year
from kronos_trading.backtest.engine import BacktestConfig, SingleAssetBacktester
from kronos_trading.data import AlpacaHistoricalDataClient
from kronos_trading.signals import (
    KronosForecastStrategy,
    KronosStrategyConfig,
    load_kronos_predictor,
)


def _parse_symbols(raw: str) -> list[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def _default_output_dir(symbol: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("backtest_runs") / f"{symbol.lower()}_{timestamp}"


def _json_default(value: object):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest a Kronos strategy on Alpaca historical bars.")
    parser.add_argument("--symbol", required=True, help="Primary symbol to trade, for example AAPL.")
    parser.add_argument("--start", required=True, help="Backtest start date/time.")
    parser.add_argument("--end", required=True, help="Backtest end date/time.")
    parser.add_argument("--timeframe", default="5Min", help="Alpaca timeframe, for example 5Min or 1Day.")
    parser.add_argument("--feed", default="iex", help="Alpaca market data feed. Paper-safe default is iex.")
    parser.add_argument("--adjustment", default="raw", help="Bar adjustment mode.")
    parser.add_argument("--benchmark-symbols", default="", help="Comma-separated benchmark symbols.")
    parser.add_argument("--model-id", default="NeoQuasar/Kronos-base", help="Kronos model ID or local path.")
    parser.add_argument(
        "--tokenizer-id",
        default="NeoQuasar/Kronos-Tokenizer-base",
        help="Kronos tokenizer ID or local path.",
    )
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--lookback", type=int, default=400, help="Historical bars passed to Kronos.")
    parser.add_argument("--max-context", type=int, default=512, help="Kronos context window.")
    parser.add_argument("--pred-len", type=int, default=1, help="Prediction horizon in bars.")
    parser.add_argument("--threshold-bps", type=float, default=20.0, help="Signal threshold in basis points.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Kronos sampling temperature.")
    parser.add_argument("--top-k", type=int, default=1, help="Top-k sampling parameter.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter.")
    parser.add_argument("--sample-count", type=int, default=1, help="Number of sampled forecast paths.")
    parser.add_argument("--allow-short", action="store_true", help="Allow short signals in backtest.")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting equity.")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Transaction fee assumption in basis points.")
    parser.add_argument("--slippage-bps", type=float, default=1.0, help="Slippage assumption in basis points.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to backtest_runs/<symbol>_<timestamp>/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    symbol = args.symbol.upper()
    benchmark_symbols = _parse_symbols(args.benchmark_symbols)
    if not benchmark_symbols and symbol != "SPY":
        benchmark_symbols = ["SPY"]

    client = AlpacaHistoricalDataClient()
    print(f"Loading primary bars for {symbol}...")
    primary_bars = client.get_symbol_bars(
        symbol=symbol,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        feed=args.feed,
        adjustment=args.adjustment,
    )

    print(f"Loading Kronos model on device: {args.device or 'auto'}")
    predictor = load_kronos_predictor(
        model_name=args.model_id,
        tokenizer_name=args.tokenizer_id,
        device=args.device,
        max_context=args.max_context,
    )
    strategy_config = KronosStrategyConfig(
        lookback=args.lookback,
        pred_len=args.pred_len,
        signal_threshold=args.threshold_bps / 10000.0,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_count=args.sample_count,
        allow_short=args.allow_short,
        verbose=False,
    )
    strategy = KronosForecastStrategy(predictor=predictor, config=strategy_config)

    periods_per_year = infer_periods_per_year(args.timeframe)
    backtest = SingleAssetBacktester(
        strategy=strategy,
        config=BacktestConfig(
            initial_capital=args.initial_capital,
            fee_bps=args.fee_bps,
            slippage_bps=args.slippage_bps,
            periods_per_year=periods_per_year,
        ),
    )
    result = backtest.run(primary_bars, symbol=symbol)

    benchmark_closes: dict[str, pd.Series] = {}
    for benchmark_symbol in tqdm(benchmark_symbols, desc="Loading benchmarks"):
        benchmark_bars = client.get_symbol_bars(
            symbol=benchmark_symbol,
            timeframe=args.timeframe,
            start=args.start,
            end=args.end,
            feed=args.feed,
            adjustment=args.adjustment,
        )
        benchmark_closes[benchmark_symbol] = benchmark_bars["close"]

    benchmarks = evaluate_benchmarks(
        strategy_index=result.results.index,
        strategy_close=primary_bars["close"],
        benchmark_closes=benchmark_closes,
        initial_capital=args.initial_capital,
        periods_per_year=periods_per_year,
        allow_short=args.allow_short,
    )

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(symbol)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.results.to_csv(output_dir / "strategy_results.csv")
    result.trades.to_csv(output_dir / "strategy_trades.csv", index=False)
    benchmarks.curves.to_csv(output_dir / "benchmark_curves.csv")

    summary = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "periods_per_year": periods_per_year,
        "strategy_metrics": result.metrics,
        "benchmark_metrics": benchmarks.metrics,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)

    print(f"Saved outputs to {output_dir}")
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
