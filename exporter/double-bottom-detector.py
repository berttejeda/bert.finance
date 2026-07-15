#!/usr/bin/env python

import datetime
import yfinance as yf

# Signal states
SIGNAL_NONE = "NONE"
SIGNAL_APPROACHING = "APPROACHING"
SIGNAL_ACTIVE = "ACTIVE"
SIGNAL_COMPLETED = "COMPLETED"


def detect_double_bottom(
    ticker_symbol,
    timeframe_days=180,
    low_tolerance=0.05,
    rebound_pct=0.03,
    bounce_confirm_days=3,
    min_bars_between_lows=10,
    max_bars_neckline_to_end=100,
    neckline_max_pct=0.75,
    max_rebound_pct=0.30,
):
    """Detect double-bottom pattern and classify current signal state.

    Signal states:
      - ACTIVE: Price hit the second low and is bouncing up, still below the
        neckline.  Best entry — ride the upswing toward the neckline.
      - COMPLETED: Price already broke above the neckline.  Pattern confirmed
        but most of the easy profit has been captured.
      - APPROACHING: Price is declining toward the first low but hasn't
        formed the second low yet.  Higher risk — bounce not confirmed.
      - NONE: No double-bottom pattern detected.

    :param ticker_symbol: Str, the stock ticker (e.g., 'MSFT')
    :param timeframe_days: Int, lookback period in calendar days
    :param low_tolerance: Float, max % difference between the two lows
        (0.05 = 5%)
    :param rebound_pct: Float, minimum % the middle peak must rise above the
        first low to qualify as a valid neckline (0.03 = 3%)
    :param bounce_confirm_days: Int, number of recent trading days to check
        for upward movement when confirming the bounce off the second low
    :param min_bars_between_lows: Int, minimum bars separating the two lows
    :param max_bars_neckline_to_end: Int, max bars from neckline to end
    :param neckline_max_pct: Float, neckline must occur within this fraction
        of data (0.75 = first 75%)
    """
    # 1. Fetch data
    ticker = yf.Ticker(ticker_symbol)
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=timeframe_days)
    df = ticker.history(start=start_date, end=end_date)

    if df.empty or len(df) < 30:
        print(f"Not enough data for {ticker_symbol}.")
        return SIGNAL_NONE

    current_price = float(df["Close"].iloc[-1])

    # 2. Find the first low — search the first 60% of bars to leave room
    #    for rebound + second low + bounce.
    cutoff = int(len(df) * 0.60)
    early_df = df.iloc[:cutoff]

    first_low_price = float(early_df["Low"].min())
    first_low_idx = early_df["Low"].idxmin()
    first_low_pos = df.index.get_loc(first_low_idx)

    # 3. Find the middle peak (neckline) — highest point between the first
    #    low and a constrained boundary (first 75% of bars).
    peak_start = first_low_pos + 1
    peak_end = min(int(len(df) * neckline_max_pct),
                   len(df) - bounce_confirm_days)
    if peak_start >= peak_end:
        print(f"Not enough data after the first low for {ticker_symbol}.")
        return SIGNAL_NONE

    peak_window = df.iloc[peak_start:peak_end]
    neckline = float(peak_window["High"].max())
    neckline_idx = peak_window["High"].idxmax()
    neckline_pos = df.index.get_loc(neckline_idx)

    # Validate the rebound is meaningful
    rebound = (neckline - first_low_price) / first_low_price
    if rebound < rebound_pct:
        print(
            f"No meaningful rebound for {ticker_symbol} "
            f"({rebound:.1%} < {rebound_pct:.1%} minimum)."
        )
        return SIGNAL_NONE

    # Reject necklines unreasonably far above the troughs (rally/crash)
    if rebound > max_rebound_pct:
        print(
            f"Neckline too high for {ticker_symbol} "
            f"({rebound:.1%} > {max_rebound_pct:.1%} maximum)."
        )
        return SIGNAL_NONE

    # 4. Find the second low — lowest point AFTER the neckline
    post_neckline = df.iloc[neckline_pos + 1:]
    if post_neckline.empty:
        return SIGNAL_NONE

    # Neckline must not be too far from end (prevents ultra-long formations)
    if len(post_neckline) > max_bars_neckline_to_end:
        return SIGNAL_NONE

    second_low_price = float(post_neckline["Low"].min())
    second_low_idx = post_neckline["Low"].idxmin()
    second_low_pos = df.index.get_loc(second_low_idx)

    # The two lows must be within tolerance of each other
    pct_diff = abs(first_low_price - second_low_price) / first_low_price
    if pct_diff > low_tolerance:
        # Check if price is still declining toward the first low (approaching)
        if current_price > first_low_price and \
                current_price < neckline and \
                current_price <= first_low_price * (1 + low_tolerance * 2):
            signal = SIGNAL_APPROACHING
            upside_to_neckline = (neckline - current_price) / current_price
            _print_report(
                ticker_symbol, signal, current_price, first_low_price,
                first_low_idx, second_low_price, second_low_idx,
                neckline, neckline_idx, upside_to_neckline, pct_diff,
            )
            return signal
        return SIGNAL_NONE

    # Enforce minimum separation between the two troughs
    if (second_low_pos - first_low_pos) < min_bars_between_lows:
        return SIGNAL_NONE

    # Validate meaningful pullback from neckline to second low
    pullback = (neckline - second_low_price) / neckline
    if pullback < rebound_pct:
        return SIGNAL_NONE

    # 5. Classify the signal
    upside_to_neckline = (neckline - current_price) / current_price

    if current_price >= neckline:
        # Price broke above neckline — pattern completed
        signal = SIGNAL_COMPLETED
    elif current_price > second_low_price:
        # Price is above the second low but below the neckline.
        # Confirm it's bouncing UP (not still falling).
        recent = df["Close"].iloc[-bounce_confirm_days:]
        is_bouncing = float(recent.iloc[-1]) > float(recent.iloc[0])

        if is_bouncing:
            signal = SIGNAL_ACTIVE
        else:
            signal = SIGNAL_APPROACHING
    else:
        signal = SIGNAL_APPROACHING

    _print_report(
        ticker_symbol, signal, current_price, first_low_price,
        first_low_idx, second_low_price, second_low_idx,
        neckline, neckline_idx, upside_to_neckline, pct_diff,
    )
    return signal


def _print_report(
    ticker, signal, current_price, first_low, first_low_idx,
    second_low, second_low_idx, neckline, neckline_idx,
    upside_to_neckline, trough_diff_pct,
):
    """Pretty-print the analysis."""
    icons = {
        SIGNAL_ACTIVE: "🟢",
        SIGNAL_COMPLETED: "✅",
        SIGNAL_APPROACHING: "🟡",
        SIGNAL_NONE: "⚪",
    }
    icon = icons.get(signal, "")

    print(f"\n{'='*55}")
    print(f" {icon}  {ticker}  —  Signal: {signal}")
    print(f"{'='*55}")
    print(f" Current Price:    ${current_price:.2f}")
    print(
        f" First Low:        ${first_low:.2f}  "
        f"({first_low_idx.strftime('%Y-%m-%d')})"
    )
    print(
        f" Second Low:       ${second_low:.2f}  "
        f"({second_low_idx.strftime('%Y-%m-%d')})  "
        f"[diff: {trough_diff_pct:.1%}]"
    )
    print(
        f" Neckline:         ${neckline:.2f}  "
        f"({neckline_idx.strftime('%Y-%m-%d')})"
    )

    if signal == SIGNAL_ACTIVE:
        print(
            f"\n 🚀 Upside to neckline: {upside_to_neckline:.1%}  "
            f"(${neckline - current_price:.2f})"
        )
        print(" ➡️  Price is bouncing off the second low — best entry window.")
    elif signal == SIGNAL_COMPLETED:
        overshoot = (current_price - neckline) / neckline
        print(f"\n ℹ️  Price is {overshoot:.1%} above the neckline.")
        print(" ➡️  Pattern confirmed but most upside has been captured.")
    elif signal == SIGNAL_APPROACHING:
        print(
            f"\n ⏳ Upside to neckline: {upside_to_neckline:.1%}  "
            f"(${neckline - current_price:.2f})"
        )
        print(" ➡️  Price is declining toward support — bounce not confirmed.")


# --- CLI ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect double-bottom patterns and classify signal state."
    )
    parser.add_argument(
        "--ticker", "-t", nargs="+", required=True,
        help="One or more stock tickers (e.g., --ticker MSFT AAPL)"
    )
    parser.add_argument(
        "--days", "-d", type=int, default=180,
        help="Lookback period in calendar days (default: 180)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Max %% difference between the two lows (default: 0.05)"
    )
    parser.add_argument(
        "--rebound", type=float, default=0.03,
        help="Min %% rebound to qualify as neckline (default: 0.03)"
    )
    args = parser.parse_args()

    for t in args.ticker:
        detect_double_bottom(
            t,
            timeframe_days=args.days,
            low_tolerance=args.tolerance,
            rebound_pct=args.rebound,
        )