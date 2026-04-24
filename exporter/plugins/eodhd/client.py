"""Python client for the EODHD API (fundamentals, historical prices, real-time)."""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("plugin.eodhd.client")

BASE_URL = "https://eodhd.com/api"


class EodhdClient:
    """Thin wrapper around eodhd.com REST endpoints."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("EODHD API key is required")
        self.api_key = api_key
        self.session = requests.Session()
        self.session.params = {"api_token": self.api_key, "fmt": "json"}  # type: ignore[assignment]

    def get_fundamentals(
        self, ticker: str, exchange: str = "US"
    ) -> Dict[str, Any]:
        """Fetch full fundamentals for a single ticker.

        Returns dict with keys: General, Highlights, Valuation,
        SharesStats, Technicals, SplitsDividends, Earnings, Financials, etc.
        """
        url = f"{BASE_URL}/fundamentals/{ticker}.{exchange}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_bulk_fundamentals(
        self,
        exchange: str = "US",
        symbols: Optional[List[str]] = None,
        offset: int = 0,
        limit: int = 500,
    ) -> List[Dict[str, Any]]:
        """Fetch fundamentals in bulk for an exchange.

        Args:
            exchange: Exchange code (e.g. 'US').
            symbols: Optional list of tickers to filter.
            offset: Pagination offset.
            limit: Max records per request (API max 500).

        Returns:
            List of fundamental dicts.
        """
        url = f"{BASE_URL}/bulk-fundamentals/{exchange}"
        params: Dict[str, Any] = {"offset": offset, "limit": limit, "version": "1.2"}
        if symbols:
            params["symbols"] = ",".join(f"{s}.{exchange}" for s in symbols)
        resp = self.session.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def get_historical_prices(
        self,
        ticker: str,
        exchange: str = "US",
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        period: str = "d",
    ) -> List[Dict[str, Any]]:
        """Fetch historical EOD price data.

        Args:
            ticker: Symbol (e.g. 'AAPL').
            exchange: Exchange code.
            date_from: Start date 'YYYY-MM-DD'.
            date_to: End date 'YYYY-MM-DD'.
            period: 'd' (daily), 'w' (weekly), 'm' (monthly).

        Returns:
            List of dicts with date, open, high, low, close, adjusted_close, volume.
        """
        url = f"{BASE_URL}/eod/{ticker}.{exchange}"
        params: Dict[str, Any] = {"period": period}
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_real_time_price(
        self, ticker: str, exchange: str = "US"
    ) -> Dict[str, Any]:
        """Fetch real-time / delayed price quote."""
        url = f"{BASE_URL}/real-time/{ticker}.{exchange}"
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_exchange_symbols(self, exchange: str = "US") -> List[Dict[str, Any]]:
        """List all symbols on an exchange."""
        url = f"{BASE_URL}/exchange-symbol-list/{exchange}"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
