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


def calc_double_bottom(df, low_tolerance=0.05, rebound_pct=0.03,
                        bounce_confirm_days=3, lookback=180):
    """Detect a double-bottom pattern and classify the current signal state.

    Signal states (stored as integers for InfluxDB compatibility):
      0 = NONE — no pattern detected
      1 = ACTIVE — bouncing off the second low, still below neckline;
          best entry window to ride the upswing
      2 = COMPLETED — price broke above the neckline; pattern confirmed but
          most easy upside has been captured
      3 = APPROACHING — price is declining toward support but the bounce off
          the second low is not yet confirmed

    Strategy:
    1. Find the first significant low in the EARLIER 60% of bars.
    2. Validate a meaningful rebound (neckline) occurred after it.
    3. Find the second low AFTER the neckline.
    4. Classify based on current price relative to second low & neckline.

    Args:
        df: DataFrame with 'Low', 'High', and 'Close' columns.
        low_tolerance: Max % difference between the two lows (0.05 = 5%).
        rebound_pct: Minimum % rebound from first low to qualify as neckline.
        bounce_confirm_days: Days to check for upward movement confirming
                            the bounce off the second low.
        lookback: Number of bars to scan for the pattern.

    Returns:
        Dict with:
          'double_bottom' (int 0-3),
          'double_bottom_neckline' (float or None),
          'double_bottom_first_low' (float or None),
          'double_bottom_first_low_date' (str ISO date or None),
          'double_bottom_second_low' (float or None),
          'double_bottom_second_low_date' (str ISO date or None).
    """
    empty = {
        "double_bottom": 0,
        "double_bottom_neckline": 0.0,
        "double_bottom_first_low": 0.0,
        "double_bottom_first_low_date": "",
        "double_bottom_second_low": 0.0,
        "double_bottom_second_low_date": "",
    }

    if not {"Close", "Low", "High"}.issubset(df.columns):
        return empty

    if len(df) < 30:
        return empty

    # Trim to lookback window
    data = df.iloc[-lookback:]
    current_price = float(data["Close"].iloc[-1])

    # Reject invalid price data
    if current_price <= 0:
        return empty

    # 1. Find the first low in the early 60% of bars
    cutoff = int(len(data) * 0.60)
    early = data.iloc[:cutoff]

    first_low_price = float(early["Low"].min())
    if first_low_price <= 0:
        return empty
    first_low_idx = early["Low"].idxmin()
    first_low_pos = data.index.get_loc(first_low_idx)

    # 2. Find the neckline (middle peak) — highest High between first low
    #    and the recent bounce_confirm_days bars.
    peak_start = first_low_pos + 1
    peak_end = len(data) - bounce_confirm_days
    if peak_start >= peak_end:
        return empty

    peak_window = data.iloc[peak_start:peak_end]
    neckline = float(peak_window["High"].max())
    neckline_idx = peak_window["High"].idxmax()
    neckline_pos = data.index.get_loc(neckline_idx)

    # Validate meaningful rebound
    rebound = (neckline - first_low_price) / first_low_price
    if rebound < rebound_pct:
        return empty

    # 3. Find the second low — lowest Low AFTER the neckline
    post_neckline = data.iloc[neckline_pos + 1:]
    if post_neckline.empty:
        return empty

    second_low_price = float(post_neckline["Low"].min())
    if second_low_price <= 0:
        return empty
    second_low_idx = post_neckline["Low"].idxmin()

    # The two lows must be within tolerance — no valid pattern otherwise
    pct_diff = abs(first_low_price - second_low_price) / first_low_price
    if pct_diff > low_tolerance:
        return empty

    # 4. Classify signal
    if current_price >= neckline:
        signal = 2  # COMPLETED
    elif current_price > second_low_price:
        # Check if price is bouncing UP (not still falling)
        recent = data["Close"].iloc[-bounce_confirm_days:]
        is_bouncing = float(recent.iloc[-1]) > float(recent.iloc[0])
        signal = 1 if is_bouncing else 3  # ACTIVE or APPROACHING
    else:
        signal = 3  # APPROACHING

    # Extra guard: if still above neckline but price came from below,
    # current_price must be below middle peak to be "active"
    if signal == 1 and current_price >= neckline:
        signal = 2

    return {
        "double_bottom": signal,
        "double_bottom_neckline": round(neckline, 4),
        "double_bottom_first_low": round(first_low_price, 4),
        "double_bottom_first_low_date": str(first_low_idx.date()),
        "double_bottom_second_low": round(second_low_price, 4),
        "double_bottom_second_low_date": str(second_low_idx.date()),
    }
