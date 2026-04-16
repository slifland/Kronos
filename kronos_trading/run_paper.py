from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from kronos_trading.data import AlpacaHistoricalDataClient
from kronos_trading.execution import AlpacaPaperTradingClient
from kronos_trading.execution.alpaca_broker import OrderRequest
from kronos_trading.signals import (
    KronosForecastStrategy,
    KronosStrategyConfig,
    load_kronos_predictor,
)


@dataclass
class PaperDecision:
    symbol: str
    timestamp: str
    predicted_return: float
    target_exposure: float
    last_close: float
    desired_qty: int
    current_qty: int
    delta_qty: int
    action: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Kronos-driven Alpaca paper trading loop.")
    parser.add_argument("--symbols", required=True, help="Comma-separated paper-trading symbols.")
    parser.add_argument("--timeframe", default="15Min", help="Alpaca timeframe, for example 15Min.")
    parser.add_argument("--feed", default="iex", help="Market data feed.")
    parser.add_argument("--lookback", type=int, default=400, help="Bars used as Kronos context.")
    parser.add_argument("--max-context", type=int, default=512, help="Kronos max context.")
    parser.add_argument("--pred-len", type=int, default=1, help="Forecast horizon in bars.")
    parser.add_argument("--threshold-bps", type=float, default=20.0, help="Signal threshold in basis points.")
    parser.add_argument("--sample-count", type=int, default=1, help="Forecast sample count.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Forecast temperature.")
    parser.add_argument("--top-k", type=int, default=1, help="Forecast top-k.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Forecast top-p.")
    parser.add_argument("--allow-short", action="store_true", help="Allow short exposure.")
    parser.add_argument("--portfolio-fraction", type=float, default=0.10, help="Max fraction of equity per symbol.")
    parser.add_argument("--max-notional", type=float, default=None, help="Optional max dollar exposure per symbol.")
    parser.add_argument("--min-order-notional", type=float, default=100.0, help="Skip tiny orders below this notional.")
    parser.add_argument("--poll-seconds", type=int, default=0, help="If >0, keep polling on this interval.")
    parser.add_argument("--allow-outside-market-hours", action="store_true", help="Allow orders when market is closed.")
    parser.add_argument("--cancel-open-orders-first", action="store_true", help="Cancel existing open orders before trading.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without submitting paper orders.")
    parser.add_argument("--device", default=None, help="Model device override.")
    parser.add_argument("--model-id", default="NeoQuasar/Kronos-base", help="Kronos model ID or local path.")
    parser.add_argument("--tokenizer-id", default="NeoQuasar/Kronos-Tokenizer-base", help="Tokenizer ID or local path.")
    parser.add_argument("--output-dir", default="paper_runs", help="Directory for decision logs.")
    return parser


def _parse_symbols(raw: str) -> list[str]:
    symbols = [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError("At least one symbol is required.")
    return symbols


def _timeframe_to_offset(timeframe: str) -> pd.Timedelta:
    lower = timeframe.lower()
    if lower.endswith("min"):
        return pd.Timedelta(minutes=int(lower[:-3]))
    if lower.endswith("hour"):
        return pd.Timedelta(hours=int(lower[:-4]))
    if lower.endswith("day"):
        return pd.Timedelta(days=int(lower[:-3]))
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _estimate_calendar_buffer_days(timeframe: str, lookback: int) -> int:
    lower = timeframe.lower()
    if lower.endswith("min"):
        minutes = int(lower[:-3])
        trading_minutes = max(lookback * minutes, 390)
        trading_days = math.ceil(trading_minutes / 390)
        return max(7, math.ceil(trading_days * 1.8) + 5)
    if lower.endswith("hour"):
        hours = int(lower[:-4])
        trading_hours = max(lookback * hours, 7)
        trading_days = math.ceil(trading_hours / 6.5)
        return max(7, math.ceil(trading_days * 1.8) + 5)
    if lower.endswith("day"):
        days = int(lower[:-3])
        return max(30, math.ceil(lookback * days * 1.5) + 10)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _recent_window(timeframe: str, lookback: int, timezone: str = "America/New_York") -> tuple[pd.Timestamp, pd.Timestamp]:
    step = _timeframe_to_offset(timeframe)
    end = pd.Timestamp.now(tz=timezone).floor(step)
    buffer_days = _estimate_calendar_buffer_days(timeframe, lookback)
    start = end - pd.Timedelta(days=buffer_days)
    return start, end


def _future_timestamps(history_index: pd.DatetimeIndex, timeframe: str, pred_len: int) -> pd.DatetimeIndex:
    step = _timeframe_to_offset(timeframe)
    last_ts = history_index[-1]
    return pd.date_range(start=last_ts + step, periods=pred_len, freq=step)


def _current_qty(position: dict[str, object] | None) -> int:
    if not position:
        return 0
    return int(float(str(position.get("qty", "0"))))


def _target_qty(account_equity: float, last_close: float, target_exposure: float, portfolio_fraction: float, max_notional: float | None) -> int:
    notional = account_equity * portfolio_fraction
    if max_notional is not None:
        notional = min(notional, max_notional)
    raw_qty = math.floor(notional / max(last_close, 0.01))
    return int(raw_qty * target_exposure)


def _build_decision(
    symbol: str,
    signal,
    account_equity: float,
    last_close: float,
    current_qty: int,
    portfolio_fraction: float,
    max_notional: float | None,
) -> PaperDecision:
    desired_qty = _target_qty(account_equity, last_close, signal.target_exposure, portfolio_fraction, max_notional)
    delta_qty = desired_qty - current_qty
    if delta_qty > 0:
        action = "buy"
    elif delta_qty < 0:
        action = "sell"
    else:
        action = "hold"

    return PaperDecision(
        symbol=symbol,
        timestamp=pd.Timestamp.now(tz="America/New_York").isoformat(),
        predicted_return=signal.predicted_return,
        target_exposure=signal.target_exposure,
        last_close=last_close,
        desired_qty=desired_qty,
        current_qty=current_qty,
        delta_qty=delta_qty,
        action=action,
    )


def _write_decisions(output_dir: Path, decisions: list[PaperDecision]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"paper_decisions_{timestamp}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(decision) for decision in decisions], handle, indent=2)


def run_once(args, data_client, broker_client, strategy) -> list[PaperDecision]:
    clock = broker_client.get_clock()
    is_open = bool(clock.get("is_open", False))
    if not is_open and not args.allow_outside_market_hours:
        print("Market is closed. Skipping run because --allow-outside-market-hours was not set.")
        return []

    if args.cancel_open_orders_first and not args.dry_run:
        canceled = broker_client.cancel_all_orders()
        print(f"Canceled {len(canceled)} open orders before evaluation.")

    account = broker_client.get_account()
    account_equity = float(account.get("equity", 0.0))
    symbols = _parse_symbols(args.symbols)

    start, end = _recent_window(args.timeframe, args.lookback)
    decisions: list[PaperDecision] = []

    for symbol in symbols:
        bars = data_client.get_symbol_bars(
            symbol=symbol,
            timeframe=args.timeframe,
            start=start,
            end=end,
            feed=args.feed,
            use_cache=False,
        )
        if len(bars) < args.lookback:
            print(f"Skipping {symbol}: only {len(bars)} bars available, need at least {args.lookback}.")
            continue

        history = bars.tail(args.lookback).copy()
        future_timestamps = _future_timestamps(pd.DatetimeIndex(history.index), args.timeframe, args.pred_len)
        signal = strategy.generate_signal(history=history, future_timestamps=future_timestamps)

        last_close = float(history["close"].iloc[-1])
        position = broker_client.get_position(symbol)
        decision = _build_decision(
            symbol=symbol,
            signal=signal,
            account_equity=account_equity,
            last_close=last_close,
            current_qty=_current_qty(position),
            portfolio_fraction=args.portfolio_fraction,
            max_notional=args.max_notional,
        )
        decisions.append(decision)

        order_notional = abs(decision.delta_qty) * decision.last_close
        print(
            f"{decision.symbol}: action={decision.action} current_qty={decision.current_qty} "
            f"target_qty={decision.desired_qty} delta={decision.delta_qty} "
            f"pred_return={decision.predicted_return:.4%} last_close={decision.last_close:.2f}"
        )

        if decision.action == "hold":
            continue
        if order_notional < args.min_order_notional:
            print(f"Skipping {symbol}: order notional {order_notional:.2f} below minimum {args.min_order_notional:.2f}.")
            continue

        side = "buy" if decision.delta_qty > 0 else "sell"
        qty = abs(decision.delta_qty)
        if args.dry_run:
            print(f"DRY RUN: would submit {side} market order for {qty} shares of {symbol}.")
            continue

        order = broker_client.submit_order(
            order=OrderRequest(
                symbol=symbol,
                side=side,
                qty=qty,
                time_in_force="day",
                order_type="market",
                extended_hours=args.allow_outside_market_hours,
                client_order_id=f"kronos-{symbol.lower()}-{int(time.time())}",
            )
        )
        print(f"Submitted order for {symbol}: id={order.get('id')} status={order.get('status')}")

    _write_decisions(Path(args.output_dir), decisions)
    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_client = AlpacaHistoricalDataClient()
    broker_client = AlpacaPaperTradingClient()
    predictor = load_kronos_predictor(
        model_name=args.model_id,
        tokenizer_name=args.tokenizer_id,
        device=args.device,
        max_context=args.max_context,
    )
    strategy = KronosForecastStrategy(
        predictor=predictor,
        config=KronosStrategyConfig(
            lookback=args.lookback,
            pred_len=args.pred_len,
            signal_threshold=args.threshold_bps / 10000.0,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_count=args.sample_count,
            allow_short=args.allow_short,
            verbose=False,
        ),
    )

    if args.poll_seconds <= 0:
        run_once(args, data_client, broker_client, strategy)
        return 0

    print(f"Starting paper loop for {args.symbols} every {args.poll_seconds} seconds.")
    while True:
        try:
            run_once(args, data_client, broker_client, strategy)
        except Exception as exc:
            print(f"Paper loop error: {exc}")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
