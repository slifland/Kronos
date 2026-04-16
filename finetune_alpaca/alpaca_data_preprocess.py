from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune_alpaca.config import AlpacaFinetuneConfig
from kronos_trading.data import AlpacaHistoricalDataClient


def load_symbols(symbols_file: str | Path) -> list[str]:
    path = Path(symbols_file)
    if not path.exists():
        raise FileNotFoundError(f"Symbols file not found: {path}")

    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip().upper()
        if not clean or clean.startswith("#"):
            continue
        symbols.append(clean)

    deduped = list(dict.fromkeys(symbols))
    if not deduped:
        raise ValueError(f"No symbols found in {path}")
    return deduped


def _localize_range(raw_range: list[str], timezone: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(raw_range[0]).tz_localize(timezone)
    end = pd.Timestamp(raw_range[1]).tz_localize(timezone) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    return start, end


def _normalize_symbol_frame(frame: pd.DataFrame, require_complete_bars: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame

    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(normalized.index).tz_localize(None)
    normalized = normalized.rename(columns={"volume": "vol", "amount": "amt"})
    normalized = normalized[["open", "high", "low", "close", "vol", "amt"]].sort_index()

    if require_complete_bars:
        normalized = normalized.dropna()

    return normalized


def split_symbol_frame(symbol_df: pd.DataFrame, config: AlpacaFinetuneConfig) -> dict[str, pd.DataFrame]:
    train_start, train_end = _localize_range(config.train_time_range, config.timezone)
    val_start, val_end = _localize_range(config.val_time_range, config.timezone)
    test_start, test_end = _localize_range(config.test_time_range, config.timezone)

    localized_index = pd.DatetimeIndex(symbol_df.index).tz_localize(config.timezone)
    train_mask = (localized_index >= train_start) & (localized_index <= train_end)
    val_mask = (localized_index >= val_start) & (localized_index <= val_end)
    test_mask = (localized_index >= test_start) & (localized_index <= test_end)

    return {
        "train": symbol_df.loc[train_mask].copy(),
        "val": symbol_df.loc[val_mask].copy(),
        "test": symbol_df.loc[test_mask].copy(),
    }


class AlpacaFinetunePreprocessor:
    def __init__(self, config: AlpacaFinetuneConfig | None = None):
        self.config = config or AlpacaFinetuneConfig()
        self.client = AlpacaHistoricalDataClient()

    def fetch_symbol_data(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        start = pd.Timestamp(self.config.dataset_begin_time).tz_localize(self.config.timezone)
        end = pd.Timestamp(self.config.dataset_end_time).tz_localize(self.config.timezone) + pd.Timedelta(days=1)

        data_by_symbol: dict[str, pd.DataFrame] = {}
        chunk_iter = range(0, len(symbols), self.config.chunk_size)
        for chunk_start in tqdm(chunk_iter, desc="Downloading Alpaca chunks"):
            chunk = symbols[chunk_start : chunk_start + self.config.chunk_size]
            bars = self.client.get_stock_bars(
                symbols=chunk,
                timeframe=self.config.timeframe,
                start=start,
                end=end,
                adjustment=self.config.adjustment,
                feed=self.config.feed,
                timezone=self.config.timezone,
            )

            for symbol, frame in self.client.split_by_symbol(bars).items():
                normalized = _normalize_symbol_frame(frame, require_complete_bars=self.config.require_complete_bars)
                if len(normalized) >= self.config.lookback_window + self.config.predict_window + 1:
                    data_by_symbol[symbol] = normalized

        return data_by_symbol

    def prepare_dataset(self, symbols: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
        raw_data = self.fetch_symbol_data(symbols)
        dataset = {"train": {}, "val": {}, "test": {}}
        min_length = self.config.lookback_window + self.config.predict_window + 1

        for symbol in tqdm(sorted(raw_data.keys()), desc="Splitting symbols"):
            split_frames = split_symbol_frame(raw_data[symbol], self.config)
            for split_name, split_frame in split_frames.items():
                if len(split_frame) >= min_length:
                    dataset[split_name][symbol] = split_frame

        return dataset

    def save_dataset(self, dataset: dict[str, dict[str, pd.DataFrame]], symbols: list[str]) -> None:
        os.makedirs(self.config.dataset_path, exist_ok=True)
        for split_name, split_payload in dataset.items():
            with open(f"{self.config.dataset_path}/{split_name}_data.pkl", "wb") as handle:
                pickle.dump(split_payload, handle)

        metadata = {
            "symbols_requested": symbols,
            "symbols_retained": {split: sorted(payload.keys()) for split, payload in dataset.items()},
            "timeframe": self.config.timeframe,
            "feed": self.config.feed,
            "adjustment": self.config.adjustment,
            "dataset_begin_time": self.config.dataset_begin_time,
            "dataset_end_time": self.config.dataset_end_time,
            "train_time_range": self.config.train_time_range,
            "val_time_range": self.config.val_time_range,
            "test_time_range": self.config.test_time_range,
            "lookback_window": self.config.lookback_window,
            "predict_window": self.config.predict_window,
            "feature_list": self.config.feature_list,
        }
        with open(self.config.metadata_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)


if __name__ == "__main__":
    config = AlpacaFinetuneConfig()
    symbols = load_symbols(config.symbols_file)
    print(f"Loaded {len(symbols)} symbols from {config.symbols_file}")
    preprocessor = AlpacaFinetunePreprocessor(config)
    dataset = preprocessor.prepare_dataset(symbols)
    preprocessor.save_dataset(dataset, symbols)
    print(f"Saved Alpaca datasets to {config.dataset_path}")
