from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PAPER_BASE_URL = "https://paper-api.alpaca.markets"


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str
    qty: int | None = None
    notional: float | None = None
    time_in_force: str = "day"
    order_type: str = "market"
    extended_hours: bool = False
    client_order_id: str | None = None


class AlpacaPaperTradingClient:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str = PAPER_BASE_URL,
        timeout: int = 30,
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

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca paper API credentials are required. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (or APCA_API_KEY_ID / APCA_API_SECRET_KEY)."
            )

    def _request_json(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        query = ""
        if params:
            clean_params = {key: value for key, value in params.items() if value not in (None, "", [])}
            query = urlencode(clean_params)

        url = f"{self.base_url}{path}"
        if query:
            url = f"{url}?{query}"

        body = None
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        request = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca trading request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Unable to reach Alpaca trading API: {exc}") from exc

    def get_account(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/v2/account")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected account response: {payload}")
        return payload

    def get_clock(self) -> dict[str, Any]:
        payload = self._request_json("GET", "/v2/clock")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected clock response: {payload}")
        return payload

    def list_positions(self) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/v2/positions")
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected positions response: {payload}")
        return payload

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        try:
            payload = self._request_json("GET", f"/v2/positions/{symbol}")
        except RuntimeError as exc:
            if "404" in str(exc):
                return None
            raise
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected position response: {payload}")
        return payload

    def list_orders(self, status: str = "open", limit: int = 100) -> list[dict[str, Any]]:
        payload = self._request_json("GET", "/v2/orders", params={"status": status, "limit": limit})
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected orders response: {payload}")
        return payload

    def cancel_all_orders(self) -> list[dict[str, Any]]:
        payload = self._request_json("DELETE", "/v2/orders")
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected cancel response: {payload}")
        return payload

    def submit_order(self, order: OrderRequest) -> dict[str, Any]:
        payload = {
            "symbol": order.symbol,
            "side": order.side,
            "type": order.order_type,
            "time_in_force": order.time_in_force,
            "extended_hours": order.extended_hours,
        }
        if order.qty is not None:
            payload["qty"] = str(order.qty)
        if order.notional is not None:
            payload["notional"] = f"{order.notional:.2f}"
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id

        response = self._request_json("POST", "/v2/orders", payload=payload)
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected order response: {response}")
        return response

