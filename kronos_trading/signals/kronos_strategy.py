from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


def _ensure_datetime_index(
    timestamps: Sequence[pd.Timestamp],
    reference_tz: str | None = None,
) -> pd.DatetimeIndex:
    if isinstance(timestamps, pd.DatetimeIndex):
        return timestamps

    inferred_tz = reference_tz
    if inferred_tz is None:
        for value in timestamps:
            tzinfo = getattr(value, "tzinfo", None)
            if tzinfo is not None:
                inferred_tz = str(tzinfo)
                break

    parsed = pd.to_datetime(list(timestamps), utc=True)
    if not isinstance(parsed, pd.DatetimeIndex):
        parsed = pd.DatetimeIndex(parsed)

    if inferred_tz is not None:
        parsed = parsed.tz_convert(inferred_tz)
    else:
        parsed = parsed.tz_localize(None)

    return parsed


@dataclass(frozen=True)
class KronosStrategyConfig:
    lookback: int = 400
    pred_len: int = 1
    signal_threshold: float = 0.002
    temperature: float = 1.0
    top_k: int = 1
    top_p: float = 1.0
    sample_count: int = 1
    allow_short: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class SignalDecision:
    decision_timestamp: pd.Timestamp
    forecast_timestamp: pd.Timestamp
    target_exposure: float
    predicted_return: float
    forecast_close: float
    reference_close: float


class KronosForecastStrategy:
    def __init__(self, predictor: Any, config: KronosStrategyConfig | None = None) -> None:
        self.predictor = predictor
        self.config = config or KronosStrategyConfig()

    def generate_signal(
        self,
        history: pd.DataFrame,
        future_timestamps: Sequence[pd.Timestamp],
    ) -> SignalDecision:
        if len(history) < self.config.lookback:
            raise ValueError(
                f"Strategy requires at least {self.config.lookback} bars, received {len(history)}."
            )

        history_index = _ensure_datetime_index(history.index)
        horizon = _ensure_datetime_index(future_timestamps, reference_tz=history_index.tz)
        if len(horizon) < self.config.pred_len:
            raise ValueError(
                f"Strategy requires {self.config.pred_len} future timestamps, received {len(horizon)}."
            )

        history = history.tail(self.config.lookback).copy()
        if "volume" not in history.columns:
            history["volume"] = 0.0
        if "amount" not in history.columns:
            history["amount"] = history["volume"] * history["close"]

        feature_frame = history[["open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
        history_index = _ensure_datetime_index(history.index)
        x_timestamp = history_index.to_series(index=range(len(history_index)))
        y_timestamp = horizon[: self.config.pred_len].to_series(index=range(self.config.pred_len))

        forecast = self.predictor.predict(
            df=feature_frame,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=self.config.pred_len,
            T=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            sample_count=self.config.sample_count,
            verbose=self.config.verbose,
        )

        forecast_close = float(forecast["close"].iloc[-1])
        reference_close = float(history["close"].iloc[-1])
        predicted_return = (forecast_close / reference_close) - 1.0

        if predicted_return > self.config.signal_threshold:
            target_exposure = 1.0
        elif self.config.allow_short and predicted_return < -self.config.signal_threshold:
            target_exposure = -1.0
        else:
            target_exposure = 0.0

        return SignalDecision(
            decision_timestamp=pd.Timestamp(history.index[-1]),
            forecast_timestamp=pd.Timestamp(y_timestamp.iloc[-1]),
            target_exposure=target_exposure,
            predicted_return=predicted_return,
            forecast_close=forecast_close,
            reference_close=reference_close,
        )


def load_kronos_predictor(
    model_name: str = "NeoQuasar/Kronos-base",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str | None = None,
    max_context: int = 512,
):
    from model import Kronos, KronosPredictor, KronosTokenizer

    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=max_context)
