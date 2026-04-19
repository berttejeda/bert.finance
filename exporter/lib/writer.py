import logging
from datetime import datetime, timezone

import math

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

from lib.indicators import calc_indicator_series

logger = logging.getLogger("writer")

# Fields that are stored as InfluxDB tags (indexed, string-only)
TAG_FIELDS = {"ticker", "industry", "bollinger_signal"}

# Fields stored as string fields (not tags, not numeric)
STRING_FIELDS = {"company_info"}

# Fields to skip (not written to InfluxDB)
SKIP_FIELDS = {"error"}


class InfluxWriter:
    def __init__(self, url, token, org, bucket):
        self.org = org
        self.bucket = bucket
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        logger.info(f"InfluxDB writer initialized: {url}, bucket={bucket}")

    def write_ticker_data(self, data):
        """Write a single ticker's data dict as an InfluxDB point.

        Args:
            data: Dict returned by fetcher.fetch_ticker_data().
        """
        if "error" in data:
            logger.warning(f"Skipping {data.get('ticker', '?')} due to error: {data['error']}")
            return

        ticker = data.get("ticker", "UNKNOWN")
        point = Point("stock_data").time(datetime.now(timezone.utc), WritePrecision.S)

        for key, value in data.items():
            if key in SKIP_FIELDS:
                continue
            if value is None:
                continue

            if key in TAG_FIELDS:
                point = point.tag(key, str(value))
            elif key in STRING_FIELDS:
                point = point.field(key, str(value))
            elif isinstance(value, (int, float)):
                point = point.field(key, float(value))
            elif isinstance(value, str):
                point = point.field(key, value)

        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        logger.info(f"Wrote data point for {ticker}")

    def write_price_history(self, ticker, history_df):
        """Write daily close prices and indicator values as individual timestamped points.

        Args:
            ticker: Stock symbol string.
            history_df: DataFrame with DatetimeIndex and 'Close'/'Volume' columns.
        """
        if history_df is None or history_df.empty:
            return

        indicators = calc_indicator_series(history_df)

        points = []
        for dt, row in history_df.iterrows():
            close = row.get("Close")
            if close is None or (isinstance(close, float) and pd.isna(close)):
                continue
            ts = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
            # Shift to noon UTC so the date displays correctly in US timezones
            ts = ts.replace(hour=12, minute=0, second=0, microsecond=0)
            p = (
                Point("price_history")
                .tag("ticker", ticker)
                .field("close", float(close))
                .time(ts, WritePrecision.S)
            )
            # Add indicator fields for this date if available
            if dt in indicators.index:
                ind_row = indicators.loc[dt]
                for col in indicators.columns:
                    val = ind_row.get(col)
                    if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                        p = p.field(col, float(val))
            points.append(p)
        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info(f"Wrote {len(points)} price_history points for {ticker}")

    # Snapshot fields that map to price_history indicator fields
    _INDICATOR_KEYS = [
        "ma_50", "ma_100", "ma_150", "ma_200",
        "rsi", "macd", "macd_signal", "macd_histogram", "vroc",
    ]

    def write_live_price(self, data):
        """Write today's live price as a price_history point.

        Uses current_price as the 'close' field and copies snapshot
        indicator values so all time-series panels extend to today.

        Args:
            data: Dict returned by fetcher.fetch_ticker_data().
        """
        if "error" in data:
            return
        ticker = data.get("ticker", "UNKNOWN")
        price = data.get("current_price")
        if price is None:
            return

        now = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0)
        p = (
            Point("price_history")
            .tag("ticker", ticker)
            .field("close", float(price))
            .time(now, WritePrecision.S)
        )
        for key in self._INDICATOR_KEYS:
            val = data.get(key)
            if val is not None and isinstance(val, (int, float)) and not (math.isnan(val) or math.isinf(val)):
                p = p.field(key, float(val))

        self.write_api.write(bucket=self.bucket, org=self.org, record=p)
        logger.info(f"Wrote live price_history point for {ticker} (close={price})")

    def write_intraday(self, ticker, intraday_df):
        """Write 1-minute OHLCV bars as individual timestamped points.

        Args:
            ticker: Stock symbol string.
            intraday_df: DataFrame with DatetimeIndex and OHLCV columns
                         from yfinance intraday download.
        """
        if intraday_df is None or intraday_df.empty:
            return

        points = []
        for dt, row in intraday_df.iterrows():
            close = row.get("Close")
            if close is None or (isinstance(close, float) and pd.isna(close)):
                continue
            ts = dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt
            p = (
                Point("price_intraday")
                .tag("ticker", ticker)
                .field("close", float(close))
                .time(ts, WritePrecision.S)
            )
            for col in ("Open", "High", "Low", "Volume"):
                val = row.get(col)
                if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    p = p.field(col.lower(), float(val))
            points.append(p)
        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info(f"Wrote {len(points)} intraday points for {ticker}")

    def write_batch(self, data_list, history_map=None, intraday_map=None):
        """Write multiple ticker data dicts and optional price history.

        Args:
            data_list: List of dicts from fetcher.fetch_ticker_data().
            history_map: Optional dict mapping ticker -> DataFrame of OHLCV history.
            intraday_map: Optional dict mapping ticker -> DataFrame of intraday bars.
        """
        for data in data_list:
            self.write_ticker_data(data)
        if history_map:
            for ticker, df in history_map.items():
                self.write_price_history(ticker, df)
        if intraday_map:
            for ticker, df in intraday_map.items():
                self.write_intraday(ticker, df)
        for data in data_list:
            self.write_live_price(data)
        logger.info(f"Batch write complete: {len(data_list)} tickers")

    def close(self):
        """Close the InfluxDB client connection."""
        self.client.close()
        logger.info("InfluxDB connection closed")
