import logging
import math
import time

import pandas as pd
import yfinance as yf

from lib.fscore import calc_fscore
from lib.indicators import (
    calc_bollinger_signal,
    calc_macd,
    calc_moving_averages,
    calc_rsi,
    calc_vroc,
)
from lib.piotroski import calc_piotroski_score

logger = logging.getLogger("fetcher")


def _safe_info_get(info, key, default=None):
    """Safely retrieve a value from yfinance info dict."""
    val = info.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return default
    return val


def _get_implied_volatility(ticker_obj):
    """Extract IV from nearest-expiry ATM call option.

    Returns:
        IV as a percentage (float) or None.
    """
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None
        chain = ticker_obj.option_chain(expirations[0])
        calls = chain.calls
        if calls.empty:
            return None
        current_price = _safe_info_get(ticker_obj.info, "currentPrice") or \
                        _safe_info_get(ticker_obj.info, "regularMarketPrice")
        if current_price is None:
            return None
        calls = calls.copy()
        calls["distance"] = (calls["strike"] - current_price).abs()
        atm = calls.loc[calls["distance"].idxmin()]
        iv = atm.get("impliedVolatility")
        if iv is not None and not (isinstance(iv, float) and math.isnan(iv)):
            return round(iv * 100, 2)
    except Exception as e:
        logger.warning(f"Could not fetch IV for {ticker_obj.ticker}: {e}")
    return None


def fetch_ticker_data(ticker, history_df, delay=2):
    """Build the complete data dict for a single ticker.

    Args:
        ticker: Stock symbol string (e.g. 'AAPL').
        history_df: DataFrame of daily OHLCV history for this ticker
                    (from a batch yf.download call), indexed by date.
        delay: Seconds to sleep after the per-ticker .info call to
               respect rate limits.

    Returns:
        Dict with all data fields, or a dict with 'ticker' and 'error'
        on failure.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        time.sleep(delay)

        current_price = _safe_info_get(info, "currentPrice") or \
                        _safe_info_get(info, "regularMarketPrice")

        market_cap = _safe_info_get(info, "marketCap")

        mas = calc_moving_averages(history_df)
        rsi = calc_rsi(history_df)
        macd = calc_macd(history_df)
        vroc = calc_vroc(history_df)
        bollinger = calc_bollinger_signal(history_df)
        iv = _get_implied_volatility(ticker_obj)

        piotroski = calc_piotroski_score(ticker_obj)
        fscore = calc_fscore(ticker_obj)

        data = {
            "ticker": ticker,
            "company_info": _safe_info_get(info, "longBusinessSummary", ""),
            "current_price": current_price,
            "market_cap": market_cap,
            "industry": _safe_info_get(info, "industry", ""),
            "ma_50": mas.get("ma_50"),
            "ma_100": mas.get("ma_100"),
            "ma_150": mas.get("ma_150"),
            "ma_200": mas.get("ma_200"),
            "week_52_high": _safe_info_get(info, "fiftyTwoWeekHigh"),
            "week_52_low": _safe_info_get(info, "fiftyTwoWeekLow"),
            "rsi": rsi,
            "macd": macd.get("macd"),
            "macd_signal": macd.get("macd_signal"),
            "macd_histogram": macd.get("macd_histogram"),
            "vroc": vroc,
            "bollinger_signal": bollinger.get("bollinger_signal"),
            "bollinger_upper": bollinger.get("bollinger_upper"),
            "bollinger_lower": bollinger.get("bollinger_lower"),
            "pe_ratio": _safe_info_get(info, "trailingPE"),
            "iv": iv,
            "piotroski_score": piotroski.get("piotroski_score"),
            "fscore": fscore.get("fscore"),
        }
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def batch_download_history(tickers, period="1y"):
    """Download daily OHLCV history for all tickers in one call.

    Args:
        tickers: List of ticker symbol strings.
        period: yfinance period string (e.g. '1y').

    Returns:
        Dict mapping ticker -> DataFrame with columns
        ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    logger.info(f"Batch downloading {period} history for {len(tickers)} tickers")
    raw = yf.download(tickers, period=period, group_by="ticker", threads=True)

    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.dropna(subset=["Close"])
            result[t] = df
        except Exception as e:
            logger.warning(f"No history data for {t}: {e}")
    return result
