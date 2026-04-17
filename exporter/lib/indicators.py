import pandas as pd


def calc_moving_averages(df, windows=(50, 100, 150, 200)):
    """Calculate simple moving averages for the given windows.

    Args:
        df: DataFrame with a 'Close' column indexed by date.
        windows: Tuple of MA window sizes.

    Returns:
        Dict mapping e.g. 'ma_50' -> latest MA value (float), or None if
        insufficient data.
    """
    result = {}
    for w in windows:
        col = f"ma_{w}"
        if len(df) >= w:
            ma_series = df["Close"].rolling(window=w).mean()
            result[col] = round(ma_series.iloc[-1], 4)
        else:
            result[col] = None
    return result


def calc_rsi(df, window=14):
    """Wilder-smoothed RSI.

    Args:
        df: DataFrame with a 'Close' column.
        window: Look-back period.

    Returns:
        Latest RSI value (float) or None.
    """
    if len(df) < window + 1:
        return None

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi.iloc[-1], 2)


def calc_macd(df, fast=12, slow=26, signal=9):
    """MACD line, signal line, and histogram.

    Args:
        df: DataFrame with a 'Close' column.

    Returns:
        Dict with 'macd', 'macd_signal', 'macd_histogram' (floats or None).
    """
    if len(df) < slow + signal:
        return {"macd": None, "macd_signal": None, "macd_histogram": None}

    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "macd": round(macd_line.iloc[-1], 4),
        "macd_signal": round(signal_line.iloc[-1], 4),
        "macd_histogram": round(histogram.iloc[-1], 4),
    }


def calc_vroc(df, period=14):
    """Volume Rate of Change.

    Args:
        df: DataFrame with a 'Volume' column.
        period: Look-back period.

    Returns:
        Latest VROC percentage (float) or None.
    """
    if len(df) < period + 1:
        return None

    vroc = ((df["Volume"] - df["Volume"].shift(period)) / df["Volume"].shift(period)) * 100
    val = vroc.iloc[-1]
    if pd.isna(val):
        return None
    return round(val, 2)


def calc_indicator_series(df):
    """Compute full daily series for all indicators.

    Args:
        df: DataFrame with 'Close' and 'Volume' columns indexed by date.

    Returns:
        DataFrame indexed by date with columns: ma_50, ma_100, ma_150, ma_200,
        rsi, macd, macd_signal, macd_histogram, vroc.  NaN rows are dropped.
    """
    result = pd.DataFrame(index=df.index)

    # Moving averages
    for w in (50, 100, 150, 200):
        if len(df) >= w:
            result[f"ma_{w}"] = df["Close"].rolling(window=w).mean()

    # RSI (Wilder-smoothed)
    if len(df) >= 15:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        result["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    if len(df) >= 35:
        ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = macd_line - signal_line

    # VROC
    if len(df) >= 15 and "Volume" in df.columns:
        result["vroc"] = (
            (df["Volume"] - df["Volume"].shift(14)) / df["Volume"].shift(14)
        ) * 100

    return result


def calc_bollinger_signal(df, window=20, std_multiplier=2):
    """Bollinger Band signal: Buy / Hold / Sell.

    Args:
        df: DataFrame with a 'Close' column.
        window: SMA window for middle band.
        std_multiplier: Number of standard deviations for bands.

    Returns:
        Dict with 'bollinger_signal', 'bollinger_upper', 'bollinger_lower'.
    """
    if len(df) < window:
        return {
            "bollinger_signal": None,
            "bollinger_upper": None,
            "bollinger_lower": None,
        }

    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    upper = sma + (std_multiplier * std)
    lower = sma - (std_multiplier * std)

    close = df["Close"].iloc[-1]
    upper_val = upper.iloc[-1]
    lower_val = lower.iloc[-1]

    if close < lower_val:
        signal = "Buy"
    elif close > upper_val:
        signal = "Sell"
    else:
        signal = "Hold"

    return {
        "bollinger_signal": signal,
        "bollinger_upper": round(upper_val, 4),
        "bollinger_lower": round(lower_val, 4),
    }
