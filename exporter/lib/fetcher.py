import logging
import math
import time

import pandas as pd
import yfinance as yf

from lib.indicators import (
    calc_bollinger_signal,
    calc_double_bottom,
    calc_macd,
    calc_moving_averages,
    calc_rsi,
    calc_vroc,
)
from lib.piotroski import calc_piotroski_score

logger = logging.getLogger("fetcher")


def _get_earnings_calendar(ticker_obj):
    """Extract upcoming earnings date and consensus estimates.

    Uses ``Ticker.calendar`` which returns a dict with keys like
    ``Earnings Date``, ``Earnings Average``, ``Revenue Average``, etc.

    Returns:
        Dict with available fields; empty dict on failure.
    """
    result = {}
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return result

        # Newer yfinance returns a plain dict
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if dates:
                first = pd.Timestamp(dates[0])
                result["next_earnings_date"] = first.isoformat()
                delta = first.tz_localize(None) - pd.Timestamp.now()
                result["days_until_earnings"] = max(int(delta.days), 0)
                if len(dates) > 1 and dates[1] != dates[0]:
                    result["next_earnings_date_end"] = pd.Timestamp(dates[1]).isoformat()

            for src, dst in [
                ("Earnings Average", "earnings_estimate_avg"),
                ("Earnings Low", "earnings_estimate_low"),
                ("Earnings High", "earnings_estimate_high"),
                ("Revenue Average", "revenue_estimate_avg"),
            ]:
                val = cal.get(src)
                if val is not None and not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
                    result[dst] = val

        # Older yfinance may return a DataFrame
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                raw = cal.loc["Earnings Date"].iloc[0] if hasattr(cal.loc["Earnings Date"], "iloc") else cal.loc["Earnings Date"]
                if pd.notna(raw):
                    first = pd.Timestamp(raw)
                    result["next_earnings_date"] = first.isoformat()
                    delta = first.tz_localize(None) - pd.Timestamp.now()
                    result["days_until_earnings"] = max(int(delta.days), 0)

    except Exception as e:
        logger.warning(f"Could not fetch earnings calendar for {ticker_obj.ticker}: {e}")
    return result


def _get_earnings_price_changes(ticker_obj, history_df):
    """Compute price changes around historical earnings dates.

    For each past earnings date, finds the close price on the last trading
    day before and the first trading day after, then calculates the
    percentage change.

    Args:
        ticker_obj: yfinance Ticker object.
        history_df: DataFrame with DatetimeIndex and 'Close' column.

    Returns:
        List of dicts with keys: report_date, price_before, price_after,
        pct_change.  Empty list on failure.
    """
    results = []
    try:
        earnings_dates = ticker_obj.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            logger.debug(f"{ticker_obj.ticker}: no earnings_dates available")
            return results

        if history_df is None or history_df.empty:
            logger.debug(f"{ticker_obj.ticker}: no history_df for earnings price changes")
            return results

        # Normalize history index to tz-naive dates for comparison
        hist = history_df[["Close"]].copy()
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert(None)
        hist = hist.sort_index()
        hist = hist.dropna(subset=["Close"])

        logger.debug(
            f"{ticker_obj.ticker}: earnings_dates has {len(earnings_dates)} entries, "
            f"history covers {hist.index.min()} to {hist.index.max()}"
        )

        for dt in earnings_dates.index:
            # Normalize earnings date to tz-naive midnight
            ts = pd.Timestamp(dt)
            if ts.tz is not None:
                ts = ts.tz_convert(None)
            report_date = ts.normalize()

            # Find last trading day before the report
            before_mask = hist.index < report_date
            if not before_mask.any():
                continue
            price_before = float(hist.loc[before_mask, "Close"].iloc[-1])

            # Find first trading day after the report
            after_mask = hist.index > report_date
            if not after_mask.any():
                continue
            price_after = float(hist.loc[after_mask, "Close"].iloc[0])

            pct_change = round(((price_after - price_before) / price_before) * 100, 2)

            results.append({
                "report_date": report_date,
                "price_before": round(price_before, 2),
                "price_after": round(price_after, 2),
                "pct_change": pct_change,
            })

        logger.debug(f"{ticker_obj.ticker}: computed {len(results)} earnings price changes")

    except Exception as e:
        logger.warning(f"Could not compute earnings price changes for {ticker_obj.ticker}: {e}")
    return results


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
        double_bottom = calc_double_bottom(history_df)

        piotroski = calc_piotroski_score(ticker_obj)
        earnings = _get_earnings_calendar(ticker_obj)
        earnings_price_changes = _get_earnings_price_changes(ticker_obj, history_df)

        bid = _safe_info_get(info, "bid")
        ask = _safe_info_get(info, "ask")
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            bid_ask_spread = round(ask - bid, 4)
            midpoint = (bid + ask) / 2
            bid_ask_spread_pct = round((bid_ask_spread / midpoint) * 100, 4) if midpoint else None
        else:
            bid_ask_spread = None
            bid_ask_spread_pct = None

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
            "bid": bid,
            "ask": ask,
            "bid_ask_spread": bid_ask_spread,
            "bid_ask_spread_pct": bid_ask_spread_pct,
            "next_earnings_date": earnings.get("next_earnings_date"),
            "next_earnings_date_end": earnings.get("next_earnings_date_end"),
            "days_until_earnings": earnings.get("days_until_earnings"),
            "earnings_estimate_avg": earnings.get("earnings_estimate_avg"),
            "earnings_estimate_low": earnings.get("earnings_estimate_low"),
            "earnings_estimate_high": earnings.get("earnings_estimate_high"),
            "revenue_estimate_avg": earnings.get("revenue_estimate_avg"),
            "earnings_price_changes": earnings_price_changes,
            "double_bottom": double_bottom.get("double_bottom"),
            "double_bottom_neckline": double_bottom.get("double_bottom_neckline"),
            "double_bottom_trough": double_bottom.get("double_bottom_trough"),
        }
        return data

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return {"ticker": ticker, "error": str(e)}


def batch_download_intraday(tickers, period="1d", interval="1m"):
    """Download intraday OHLCV bars for all tickers in one call.

    Args:
        tickers: List of ticker symbol strings.
        period: yfinance period string (e.g. '1d', '5d', '7d').
        interval: Bar interval string (e.g. '1m', '2m', '5m').

    Returns:
        Dict mapping ticker -> DataFrame with columns
        ['Open', 'High', 'Low', 'Close', 'Volume'] and a
        timezone-aware DatetimeIndex.
    """
    logger.info(f"Batch downloading {period} intraday ({interval}) for {len(tickers)} tickers")
    raw = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True)

    if raw.empty:
        logger.warning("Intraday download returned no data")
        return {}

    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
            else:
                df = raw[t].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.dropna(subset=["Close"])
            if not df.empty:
                result[t] = df
        except Exception as e:
            logger.warning(f"No intraday data for {t}: {e}")
    return result


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
