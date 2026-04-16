from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_BASE_URL = "https://data.alpaca.markets/v2"
DEFAULT_TIMEZONE = "America/New_York"


def _format_timestamp(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.isoformat().replace("+00:00", "Z")


def _cache_key(path: str, params: dict[str, object]) -> str:
    payload = json.dumps({"path": path, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AlpacaHistoricalDataClient:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("ALPACA_API_KEY_ID")
            or os.getenv("APCA_API_KEY_ID")
        )
        self.secret_key = (
            secret_key
            or os.getenv("ALPACA_API_SECRET_KEY")
            or os.getenv("APCA_API_SECRET_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.cache_dir = Path(cache_dir or ".cache/alpaca")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API credentials are required. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (or APCA_API_KEY_ID / APCA_API_SECRET_KEY)."
            )

    def _request_json(self, path: str, params: dict[str, object]) -> dict[str, object]:
        clean_params = {key: value for key, value in params.items() if value not in (None, "", [])}
        query = urlencode(clean_params)
        url = f"{self.base_url}{path}?{query}"
        request = Request(
            url,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach Alpaca market data API: {exc}") from exc

    def get_stock_bars(
        self,
        symbols: str | Sequence[str],
        timeframe: str,
        start: object,
        end: object,
        adjustment: str = "raw",
        feed: str = "iex",
        sort: str = "asc",
        limit: int = 10000,
        timezone: str = DEFAULT_TIMEZONE,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        if isinstance(symbols, str):
            symbol_list = [symbols]
        else:
            symbol_list = list(symbols)

        if not symbol_list:
            raise ValueError("At least one symbol is required.")

        params: dict[str, object] = {
            "symbols": ",".join(sorted(symbol_list)),
            "timeframe": timeframe,
            "start": _format_timestamp(start),
            "end": _format_timestamp(end),
            "adjustment": adjustment,
            "feed": feed,
            "sort": sort,
            "limit": limit,
        }

        cache_path = self.cache_dir / f"{_cache_key('/stocks/bars', params)}.csv"
        if use_cache and cache_path.exists():
            cached = pd.read_csv(cache_path, parse_dates=["timestamp"])
            return cached

        records: list[dict[str, object]] = []
        page_token: str | None = None

        while True:
            page_params = dict(params)
            if page_token:
                page_params["page_token"] = page_token

            payload = self._request_json("/stocks/bars", page_params)
            bars_by_symbol = payload.get("bars", {})
            if not isinstance(bars_by_symbol, dict):
                raise RuntimeError(f"Unexpected Alpaca response shape: {payload}")

            for symbol, bars in bars_by_symbol.items():
                for bar in bars:
                    timestamp = pd.Timestamp(bar["t"])
                    if timestamp.tzinfo is None:
                        timestamp = timestamp.tz_localize("UTC")
                    timestamp = timestamp.tz_convert(timezone)
                    volume = float(bar.get("v", 0.0))
                    close = float(bar["c"])
                    records.append(
                        {
                            "symbol": symbol,
                            "timestamp": timestamp,
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": close,
                            "volume": volume,
                            "trade_count": int(bar.get("n", 0) or 0),
                            "vwap": float(bar.get("vw", close)),
                            "amount": close * volume,
                        }
                    )

            page_token = payload.get("next_page_token")
            if not page_token:
                break

        frame = pd.DataFrame.from_records(
            records,
            columns=[
                "symbol",
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "trade_count",
                "vwap",
                "amount",
            ],
        )
        if not frame.empty:
            frame = frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache_path, index=False)

        return frame

    def get_symbol_bars(
        self,
        symbol: str,
        timeframe: str,
        start: object,
        end: object,
        adjustment: str = "raw",
        feed: str = "iex",
        sort: str = "asc",
        limit: int = 10000,
        timezone: str = DEFAULT_TIMEZONE,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        frame = self.get_stock_bars(
            symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end,
            adjustment=adjustment,
            feed=feed,
            sort=sort,
            limit=limit,
            timezone=timezone,
            use_cache=use_cache,
        )
        frame = frame[frame["symbol"] == symbol].copy()
        frame = frame.drop(columns=["symbol"]).set_index("timestamp")
        return frame

    @staticmethod
    def split_by_symbol(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if frame.empty:
            return {}

        grouped: dict[str, pd.DataFrame] = {}
        for symbol, group in frame.groupby("symbol", sort=True):
            grouped[symbol] = group.drop(columns=["symbol"]).set_index("timestamp").sort_index()
        return grouped

