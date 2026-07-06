#!/usr/bin/env python3
"""Add description fields to Grafana dashboard panels.

Usage:
    python3 add_panel_descriptions.py dashboard-all.json dashboard.json
"""

import json
import sys

DESCRIPTIONS = {
    "Company Details": (
        "Business summary, industry classification, and last-updated timestamp "
        "for each tracked stock."
    ),
    "Current Price": (
        "Latest trade price from [yfinance](https://pypi.org/project/yfinance/). "
        "Updated each export cycle."
    ),
    "P/E Ratio": (
        "[Price-to-Earnings ratio](https://www.investopedia.com/terms/p/price-earningsratio.asp) "
        "(trailing 12 months). Lower values may indicate undervaluation relative to earnings; "
        "high values suggest growth expectations or overvaluation. Negative = unprofitable."
    ),
    "Market Cap": (
        "[Market capitalization](https://www.investopedia.com/terms/m/marketcapitalization.asp) "
        "= share price × shares outstanding. "
        "Micro (<$300M), Small (<$2B), Mid (<$10B), Large (<$200B), Mega (>$200B)."
    ),
    "Piotroski Score": (
        "[Piotroski F-Score](https://www.investopedia.com/terms/p/piotroski-score.asp) (0–9). "
        "Measures financial strength across profitability, leverage, liquidity, and operating "
        "efficiency. ≥7 = strong fundamentals; ≤3 = weakness."
    ),
    "F-Score": (
        "[Piotroski F-Score](https://www.investopedia.com/terms/p/piotroski-score.asp) (0–9). "
        "Legacy duplicate of Piotroski Score — uses older data field."
    ),
    "Implied Volatility": (
        "[Implied volatility](https://www.investopedia.com/terms/i/iv.asp) (IV) from "
        "nearest-expiry ATM call option. Higher IV = market expects larger price swings. "
        "Useful for [options](https://www.investopedia.com/terms/o/option.asp) pricing "
        "and gauging uncertainty."
    ),
    "Bollinger Signal": (
        "[Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp) signal "
        "(Buy / Hold / Sell). Buy = price near lower band (potential "
        "[support](https://www.investopedia.com/terms/s/support.asp)); "
        "Sell = price near upper band (potential "
        "[resistance](https://www.investopedia.com/terms/r/resistance.asp))."
    ),
    "52-Week High": (
        "Highest trade price in the past 52 weeks. Proximity may indicate "
        "[resistance](https://www.investopedia.com/terms/r/resistance.asp) or a breakout."
    ),
    "52-Week Low": (
        "Lowest trade price in the past 52 weeks. Proximity may indicate "
        "[support](https://www.investopedia.com/terms/s/support.asp) or further downside."
    ),
    "Intraday Price (1-min)": (
        "1-minute [OHLCV](https://www.investopedia.com/terms/o/ohlcchart.asp) close prices "
        "for the most recent trading session(s)."
    ),
    "Price & Moving Averages": (
        "Daily close overlaid with [Simple Moving Averages](https://www.investopedia.com/terms/s/sma.asp) "
        "(50, 100, 150, 200-day). Price crossing above/below an MA may signal trend changes. "
        "The 200-day MA is a widely-watched long-term trend indicator."
    ),
    "RSI (14)": (
        "[Relative Strength Index](https://www.investopedia.com/terms/r/rsi.asp) (14-period). "
        "Oscillator 0–100. >70 = [overbought](https://www.investopedia.com/terms/o/overbought.asp) "
        "(potential pullback); <30 = [oversold](https://www.investopedia.com/terms/o/oversold.asp) "
        "(potential bounce). Shaded zones highlight these regions."
    ),
    "MACD": (
        "[Moving Average Convergence Divergence](https://www.investopedia.com/terms/m/macd.asp). "
        "MACD crossing above the signal line = bullish; below = bearish. "
        "Histogram bars show the gap — growing bars = strengthening momentum."
    ),
    "VROC": (
        "[Volume Rate of Change](https://www.investopedia.com/terms/v/volumepricetrend.asp) — "
        "% change in volume over a lookback period. Spikes may confirm price moves "
        "(high conviction) or signal reversals."
    ),
    "Analyst Consensus": (
        "Distribution of analyst ratings (Strong Buy → Strong Sell) from "
        "[EODHD](https://eodhd.com/). Stacked bars show the sentiment balance per ticker."
    ),
    "Earnings Surprise — EPS Actual vs Estimate": (
        "[EPS](https://www.investopedia.com/terms/e/eps.asp) actual vs analyst consensus "
        "estimate. Points above the estimate line = positive surprise (beat); below = miss."
    ),
    "Short Ratio": (
        "[Short ratio](https://www.investopedia.com/terms/s/shortinterestratio.asp) (days to cover) "
        "= short interest ÷ avg daily volume. Higher = more days for shorts to unwind, "
        "increasing [short squeeze](https://www.investopedia.com/terms/s/shortsqueeze.asp) potential."
    ),
    "Short % of Float": (
        "% of [float](https://www.investopedia.com/terms/f/float.asp) sold short. "
        ">10% = elevated; >20% = very high — bearish sentiment or squeeze risk."
    ),
    "Profit Margin": (
        "[Net profit margin](https://www.investopedia.com/terms/n/net_margin.asp) = "
        "net income ÷ revenue. Higher is better; negative = losing money."
    ),
    "Return on Equity (ROE)": (
        "[ROE](https://www.investopedia.com/terms/r/returnonequity.asp) = net income ÷ "
        "shareholders' equity. Measures how efficiently profits are generated from equity. "
        "Above 15–20% is generally strong."
    ),
    "Quarterly Earnings Growth (YoY)": (
        "Year-over-year growth in quarterly [earnings](https://www.investopedia.com/terms/e/earnings.asp). "
        "Positive = growing profits; negative = shrinking."
    ),
    "Quarterly Revenue Growth (YoY)": (
        "Year-over-year growth in quarterly [revenue](https://www.investopedia.com/terms/r/revenue.asp). "
        "Positive = top-line growth. Compare with earnings growth to assess margin trends."
    ),
    "Dividend Yield": (
        "[Dividend yield](https://www.investopedia.com/terms/d/dividendyield.asp) = "
        "annual dividend ÷ share price. >4–5% may indicate value or distress; compare within sector."
    ),
    "Valuation Multiples": (
        "Key [valuation ratios](https://www.investopedia.com/terms/v/valuationratio.asp): "
        "Trailing P/E, Forward P/E, "
        "[P/B](https://www.investopedia.com/terms/p/price-to-bookratio.asp), "
        "[P/S](https://www.investopedia.com/terms/p/price-to-salesratio.asp), "
        "[PEG](https://www.investopedia.com/terms/p/pegratio.asp), "
        "[EV/EBITDA](https://www.investopedia.com/terms/e/ev-ebitda.asp). "
        "Lower multiples vs peers may suggest undervaluation."
    ),
    "Quarterly Revenue & EBITDA": (
        "Quarterly [revenue](https://www.investopedia.com/terms/r/revenue.asp) and "
        "[EBITDA](https://www.investopedia.com/terms/e/ebitda.asp) from financial statements. "
        "Shows top-line and operational profitability trends."
    ),
    "Beta": (
        "[Beta](https://www.investopedia.com/terms/b/beta.asp) — volatility relative to "
        "the market (S&P 500). 1.0 = moves with market; >1.0 = more volatile; "
        "<1.0 = less volatile; negative = inverse."
    ),
    "Analyst Target vs Current Price": (
        "Current price vs consensus [analyst target](https://www.investopedia.com/terms/p/pricetarget.asp). "
        "Upside % = (target − current) ÷ current. Positive = analysts expect appreciation."
    ),
    "RSI - AI Analysis": (
        "AI-powered analysis of [RSI](https://www.investopedia.com/terms/r/rsi.asp) data. "
        "LLM identifies trends, overbought/oversold conditions, and actionable insights."
    ),
    "MACD AI Analysis": (
        "AI-powered analysis of [MACD](https://www.investopedia.com/terms/m/macd.asp) data. "
        "LLM interprets signal crossovers, histogram momentum, and trend direction."
    ),
    "Ask AI": (
        "Chat-style financial Q&A. Ask natural-language questions about stock data in "
        "InfluxDB — the AI generates Flux queries, executes them, and formats the answer."
    ),
}

# Partial-match fallback for dynamic titles like "Price History — AAPL (6mo)"
PARTIAL_DESCRIPTIONS = {
    "Price History": (
        "Daily close prices over the selected range. Uses the **Price History Range** "
        "dropdown (not the dashboard time picker). Useful for identifying long-term "
        "[trends](https://www.investopedia.com/terms/t/trend.asp) and "
        "[support/resistance](https://www.investopedia.com/terms/s/supportandresistance.asp) levels."
    ),
}


def add_descriptions(panels):
    count = 0
    for panel in panels:
        title = panel.get("title", "")
        if title in DESCRIPTIONS and not panel.get("description"):
            panel["description"] = DESCRIPTIONS[title]
            count += 1
        else:
            for partial, desc in PARTIAL_DESCRIPTIONS.items():
                if partial in title and not panel.get("description"):
                    panel["description"] = desc
                    count += 1
                    break
        # Recurse into row panels
        count += add_descriptions(panel.get("panels", []))
    return count


def main():
    files = sys.argv[1:] or ["dashboard-all.json", "dashboard.json"]
    for fpath in files:
        with open(fpath) as f:
            dashboard = json.load(f)

        n = add_descriptions(dashboard.get("panels", []))

        with open(fpath, "w") as f:
            json.dump(dashboard, f, indent=2)
            f.write("\n")

        print(f"{fpath}: {n} panel descriptions added")


if __name__ == "__main__":
    main()
