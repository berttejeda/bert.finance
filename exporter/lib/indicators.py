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


def calc_double_bottom(df, sma_window=20, min_distance=20, tolerance=0.05,
                        neckline_min_pct=0.03, lookback=250):
    """Detect a double-bottom reversal pattern.

    Uses the Low price (support bounces are best captured at intraday lows)
    with aggressive smoothing to identify structural troughs.  Evaluates all
    pairs of detected troughs — not just the last two — to avoid missing
    patterns when noise creates additional local minima.

    A double bottom is confirmed when:
    1. Two troughs are within ``tolerance`` of each other
    2. A neckline (peak between troughs) rises at least ``neckline_min_pct``
       above the average trough price
    3. Current price has broken above the neckline

    Args:
        df: DataFrame with 'Low' and 'Close' columns indexed by date.
        sma_window: Rolling window for noise smoothing (larger = fewer false
                    minima).
        min_distance: Minimum number of bars between troughs.
        tolerance: Max percentage difference between trough prices (0.05 = 5%).
        neckline_min_pct: Minimum neckline rise above troughs (0.03 = 3%).
        lookback: Number of bars to scan for the pattern.

    Returns:
        Dict with 'double_bottom' (0 or 1), 'double_bottom_neckline',
        and 'double_bottom_trough' (float prices or None).
    """
    empty = {
        "double_bottom": 0,
        "double_bottom_neckline": None,
        "double_bottom_trough": None,
    }

    # Use Low for trough detection (captures intraday support bounces),
    # fall back to Close if Low is unavailable.
    price_col = "Low" if "Low" in df.columns else "Close"

    min_bars = sma_window + min_distance * 3
    if len(df) < min_bars:
        return empty

    low = df[price_col].iloc[-lookback:]
    close = df["Close"].iloc[-lookback:]
    smoothed = low.rolling(window=sma_window).mean().dropna()

    if len(smoothed) < min_distance * 3:
        return empty

    vals = smoothed.values
    n = len(vals)
    half = min_distance // 2

    # Detect local minima: each point lower than its neighbourhood
    trough_idx = []
    for i in range(half, n - half):
        window = vals[max(0, i - half): i + half + 1]
        if vals[i] == window.min():
            trough_idx.append(i)

    # Enforce minimum spacing — keep the deeper trough when too close
    if len(trough_idx) >= 2:
        merged = [trough_idx[0]]
        for idx in trough_idx[1:]:
            if idx - merged[-1] >= min_distance:
                merged.append(idx)
            elif vals[idx] < vals[merged[-1]]:
                merged[-1] = idx
        trough_idx = merged

    if len(trough_idx) < 2:
        return empty

    # Evaluate ALL pairs of troughs (most recent pair first) to avoid
    # missing valid patterns when extra noise-troughs exist.
    current = float(close.iloc[-1])
    best = None

    for pair_i in range(len(trough_idx) - 1, 0, -1):
        for pair_j in range(pair_i - 1, -1, -1):
            i1, i2 = trough_idx[pair_j], trough_idx[pair_i]
            date1 = smoothed.index[i1]
            date2 = smoothed.index[i2]
            t1 = float(low.loc[date1])
            t2 = float(low.loc[date2])

            # Troughs must be at a similar level
            if abs(t1 - t2) / min(t1, t2) > tolerance:
                continue

            # Neckline — highest low between the two troughs
            between = low.loc[date1:date2]
            neckline = float(between.max())
            avg_trough = (t1 + t2) / 2.0

            # Neckline must rise meaningfully above the troughs
            if (neckline - avg_trough) / avg_trough < neckline_min_pct:
                continue

            # Pattern confirmed only when price broke above neckline
            if current < neckline:
                continue

            # Prefer the pair with the lowest trough (strongest support)
            trough_price = min(t1, t2)
            if best is None or trough_price < best["trough_price"]:
                best = {
                    "trough_price": trough_price,
                    "neckline": neckline,
                }

    if best:
        return {
            "double_bottom": 1,
            "double_bottom_neckline": round(best["neckline"], 4),
            "double_bottom_trough": round(best["trough_price"], 4),
        }

    return empty
