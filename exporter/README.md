# Stock Data Exporter → InfluxDB

Fetches stock data via `yfinance` and writes it to an InfluxDB 2.x time-series database.
A companion Grafana dashboard (`grafana/dashboard.json`) visualizes the data.

## Data Points

| Field | Source |
|---|---|
| Ticker | Config |
| Company Info | `yf.Ticker.info` → `longBusinessSummary` |
| Current Price | `yf.Ticker.info` → `currentPrice` |
| Market Cap | `yf.Ticker.info` → `marketCap` |
| Industry | `yf.Ticker.info` → `industry` |
| 50/100/150/200-MA | Computed from 1y daily closes (`pandas.rolling`) |
| 52w High / Low | `yf.Ticker.info` → `fiftyTwoWeekHigh/Low` |
| RSI (14) | Wilder-smoothed, computed from daily closes |
| MACD | EMA(12) − EMA(26), signal = EMA(9) |
| VROC | Volume Rate of Change, 14-period |
| Bollinger Signal | 20-day SMA ± 2σ → Buy/Hold/Sell |
| P/E Ratio | `yf.Ticker.info` → `trailingPE` |
| IV | Nearest-expiry ATM call implied volatility |
| Piotroski Score | 9-signal financial strength score (0-9) from annual statements |
| F-Score | Ratio-based Piotroski F-Score per Piotroski (2000) methodology |

## Prerequisites

- Python 3.10+
- InfluxDB 2.x instance with a bucket created

## Setup

```bash
cd exporter
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Set the following environment variables:

```bash
export INFLUXDB_HOST=http://localhost:8086
export INFLUXDB_TOKEN=your-influxdb-token
export INFLUXDB_ORG=your-org
export INFLUXDB_BUCKET=stocks
```

Edit `config.yaml` to customize the ticker list and settings.

### Settings Reference

| Key | Default | Description |
|---|---|---|
| `interval_minutes` | `30` | Minutes between exporter runs (loop mode) |
| `delay_between_tickers` | `2` | Seconds between per-ticker API calls |
| `history_period` | `1y` | yfinance period for daily history download |
| `intraday_period` | `5d` | yfinance period for intraday download (max `7d`) |
| `intraday_interval` | `1m` | Bar interval for intraday download |

> **Note**: Duplicate tickers in `config.yaml` are automatically deduplicated at runtime.

## Usage

### One-shot run

```bash
python exporter.py
```

### Continuous loop

```bash
python exporter.py --loop
```

### Custom config path

```bash
python exporter.py --config /path/to/config.yaml
```

## Architecture

### Write Pipeline (`lib/writer.py`)

The `InfluxWriter` class handles all writes to InfluxDB. Each call to `write_batch()` performs three steps in order:

1. **`write_ticker_data(data)`** — Writes a snapshot point to `stock_data` for each ticker with the current UTC ingestion timestamp.
2. **`write_price_history(ticker, df)`** — Writes daily historical close prices and computed indicator series to `price_history` using the trading-day date from the yfinance DataFrame index.
3. **`write_intraday(ticker, df)`** — Writes 1-minute OHLCV bars to `price_intraday` using timestamps from the yfinance intraday download.
4. **`write_live_price(data)`** — Writes today's live price and snapshot indicator values as an additional `price_history` point so time-series panels extend to the current day.

### Indicator Series (`lib/indicators.py`)

`calc_indicator_series(df)` computes full daily series for MA-50/100/150/200, RSI, MACD (line, signal, histogram), and VROC. These are written as fields alongside `close` in `price_history`, enabling time-series panels for each indicator.

## InfluxDB Schema

### `stock_data` (snapshot per run)

- **Tags**: `ticker`, `industry`, `bollinger_signal`
- **Fields**: `current_price`, `market_cap`, `ma_50`, `ma_100`, `ma_150`, `ma_200`, `week_52_high`, `week_52_low`, `rsi`, `macd`, `macd_signal`, `macd_histogram`, `vroc`, `pe_ratio`, `iv`, `piotroski_score`, `fscore`, `company_info`, `bollinger_upper`, `bollinger_lower`
- **Timestamp**: UTC collection time (second precision)

### `price_history` (daily prices & indicators)

- **Tags**: `ticker`
- **Fields**: `close`, `ma_50`, `ma_100`, `ma_150`, `ma_200`, `rsi`, `macd`, `macd_signal`, `macd_histogram`, `vroc`
- **Timestamp**: Trading-day date at noon UTC (second precision)

### `price_intraday` (1-minute OHLCV bars)

- **Tags**: `ticker`
- **Fields**: `open`, `high`, `low`, `close`, `volume`
- **Timestamp**: Original minute-level timestamp from yfinance (second precision, timezone-aware)

### Timestamp Convention

All `price_history` timestamps are set to **noon UTC (12:00:00Z)** of the trading day. This ensures dates display correctly in the Grafana dashboard regardless of the viewer's browser timezone. Midnight-UTC timestamps shift to the previous calendar day in US timezones (e.g., `2026-04-16T00:00:00Z` → April 15 8:00 PM EDT).

## Grafana Dashboard

The dashboard is stored in `grafana/dashboard.json`. A working copy with the latest manual customizations is kept in `grafana/dashboard-modified.json`.

### Duplicate Series Prevention

`bollinger_signal` is an InfluxDB **tag** on `stock_data` points. Because its value changes over time (Buy / Hold / Sell), InfluxDB creates separate series for each distinct tag combination. To prevent duplicate rows in Grafana panels, all `stock_data` snapshot queries include:

```flux
|> group(columns: ["ticker", "_field"])
|> last()
```

This collapses all series for the same ticker + field before selecting the latest value, regardless of tag variation.

### Panel Queries — Pattern Reference

**Snapshot panels** (Current Price, P/E, Market Cap, 52-Week, Piotroski, F-Score, IV, Bollinger Signal, Industry, Company Details, All Tickers table) query `stock_data`:

```flux
from(bucket: "${bucket}")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> filter(fn: (r) => r["_field"] == "<field_name>")
  |> group(columns: ["ticker", "_field"])
  |> last()
  |> map(fn: (r) => ({ticker: r.ticker, _value: r._value}))
  |> group()
```

**Time-series panels** (Price History, Price & Moving Averages, RSI, MACD, VROC) query `price_history`:

```flux
from(bucket: "${bucket}")
  |> range(start: -${price_range})
  |> filter(fn: (r) => r["_measurement"] == "price_history")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> filter(fn: (r) => r["_field"] == "close" or r["_field"] == "ma_50" ...)
```

**Intraday panel** (Intraday Price) queries `price_intraday`:

```flux
from(bucket: "${bucket}")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "price_intraday")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> filter(fn: (r) => r["_field"] == "close")
```

**Bollinger Signal** reads the tag value from the `current_price` record:

```flux
|> filter(fn: (r) => r["_field"] == "current_price")
|> group(columns: ["ticker", "_field"])
|> last()
|> map(fn: (r) => ({_value: r.bollinger_signal, ticker: r.ticker}))
|> group()
```
