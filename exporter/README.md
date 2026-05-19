# Stock Data Exporter → InfluxDB

Fetches stock data via `yfinance` and writes it to an InfluxDB 2.x time-series database.
A companion Grafana dashboard (`grafana/dashboard.json`) visualizes the data.
Extensible via a **plugin system** for additional data sources (e.g. EODHD fundamentals).

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
| Bid / Ask / Spread | `yf.Ticker.info` → `bid`, `ask`, computed spread & spread % |
| Days Until Earnings | Computed from `yf.Ticker.calendar` → next earnings date |
| Earnings Estimate (Avg/Low/High) | `yf.Ticker.calendar` → consensus EPS estimates |
| Revenue Estimate (Avg) | `yf.Ticker.calendar` → consensus revenue estimate |
| Next Earnings Date | `yf.Ticker.calendar` → ISO 8601 date string(s) |
| Earnings Price Changes | Computed from `yf.Ticker.earnings_dates` + daily close history: price 1 day before, 1 day after each report, and percentage change |

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

### Environment Variables

Environment variables can be set directly or via a `.env` file in the project root (loaded automatically via `python-dotenv`).

```bash
INFLUXDB_HOST=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token
INFLUXDB_ORG=your-org
INFLUXDB_BUCKET=stocks
```

Optional (for plugins):

```bash
EODHD_API_TOKEN=your-eodhd-api-key
```

Edit `config.yaml` to customize the ticker list, settings, and plugins.

### Settings Reference

| Key | Default | Description |
|---|---|---|
| `interval_minutes` | `30` | Minutes between exporter runs (loop mode) |
| `delay_between_tickers` | `2` | Seconds between per-ticker API calls |
| `history_period` | `1y` | yfinance period for daily history download |
| `intraday_period` | `5d` | yfinance period for intraday download (max `7d`) |
| `intraday_interval` | `1m` | Bar interval for intraday download |
| `max_retries` | `3` | Number of retry attempts on InfluxDB write timeout |
| `retry_delay` | `5` | Base delay in seconds between retries (doubles each attempt) |
| `timeout` | `30` | InfluxDB HTTP request timeout in seconds |

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

### Override tickers

```bash
python exporter.py --tickers AAPL,MSFT,GOOGL
```

### Run a specific plugin only (skip core export)

```bash
python exporter.py --plugin eodhd
```

### Debug mode

Shows InfluxDB line protocol for every point written:

```bash
python exporter.py --debug
```

### Override retry / timeout settings

```bash
python exporter.py --retries 5 --retry-delay 10 --timeout 60
```

### Stop at market close

```bash
python exporter.py --loop --run-until-market-close
```

### Stop at a specific time

```bash
python exporter.py --loop --run-until-time "4:00 PM"
python exporter.py --loop --run-until-time "16:00"
```

### CLI Reference

| Flag | Short | Description |
|---|---|---|
| `--config` | | Path to config.yaml (default: `config.yaml`) |
| `--loop` | | Run continuously at the configured interval |
| `--tickers` | `-t` | Comma-separated ticker list (overrides config) |
| `--plugin` | `-p` | Run only the named plugin (skip core export) |
| `--retries` | | Max write retries on timeout (overrides config) |
| `--retry-delay` | | Base delay in seconds between retries (overrides config) |
| `--timeout` | | InfluxDB write timeout in seconds (overrides config) |
| `--debug` | `-d` | Enable debug logging (line protocol, etc.) |
| `--run-until-market-close` | | Stop the loop at US market close (4:00 PM Eastern) |
| `--run-until-time` | | Stop the loop at a given local time (e.g. `4:00 PM`, `16:00`) |

## Architecture

### Write Pipeline (`lib/writer.py`)

The `InfluxWriter` class handles all writes to InfluxDB. Each call to `write_batch()` performs five steps in order:

1. **`write_ticker_data(data)`** — Writes a snapshot point to `stock_data` for each ticker with the current UTC ingestion timestamp.
2. **`write_price_history(ticker, df)`** — Writes daily historical close prices and computed indicator series to `price_history` using the trading-day date from the yfinance DataFrame index.
3. **`write_intraday(ticker, df)`** — Writes 1-minute OHLCV bars to `price_intraday` using timestamps from the yfinance intraday download.
4. **`write_live_price(data)`** — Writes today's live price and snapshot indicator values as an additional `price_history` point so time-series panels extend to the current day.
5. **`write_earnings_price_changes(ticker, changes)`** — Writes earnings-related price change records to `earnings_price_change` with the report date as the timestamp. Data is computed by `_get_earnings_price_changes()` which cross-references `yf.Ticker.earnings_dates` with the daily close history.

All writes use `_write_with_retry()`, which retries on timeout/connection errors with exponential backoff (base delay doubles each attempt).

### Indicator Series (`lib/indicators.py`)

`calc_indicator_series(df)` computes full daily series for MA-50/100/150/200, RSI, MACD (line, signal, histogram), and VROC. These are written as fields alongside `close` in `price_history`, enabling time-series panels for each indicator.

### Plugin System

Plugins extend the exporter with additional data sources. They live in `plugins/<name>/plugin.py` and inherit from `PluginBase`.

#### How it works

1. The `plugins` section in `config.yaml` defines available plugins with `name`, `enabled`, and `args`.
2. `plugins/loader.py` discovers and loads enabled plugins at runtime.
3. After the core export completes, each enabled plugin's `run(args)` method is called.
4. The loader injects the global `tickers` and `influxdb` config into each plugin's args unless the plugin overrides them.

#### Creating a new plugin

1. Create `plugins/<name>/plugin.py` with a `Plugin` class inheriting from `PluginBase`.
2. Implement the `name` property and `run(args)` method.
3. Add an entry under `plugins:` in `config.yaml`.

```python
from plugins.base import PluginBase

class Plugin(PluginBase):
    @property
    def name(self) -> str:
        return "my_plugin"

    def run(self, args):
        tickers = args["tickers"]       # list of ticker symbols
        influx = args["influxdb"]        # dict with url, token, org, bucket
        # ... fetch data and write to InfluxDB
```

#### Built-in plugins

- **`eodhd`** — Fetches fundamentals, financial statements, and earnings from the EODHD API. Writes to `eodhd_fundamentals`, `eodhd_financials`, and `eodhd_earnings` measurements. Requires `EODHD_API_TOKEN`.

## InfluxDB Schema

### Core Measurements

#### `stock_data` (snapshot per run)

- **Tags**: `ticker`, `industry`, `bollinger_signal`
- **Fields**: `current_price`, `market_cap`, `ma_50`, `ma_100`, `ma_150`, `ma_200`, `week_52_high`, `week_52_low`, `rsi`, `macd`, `macd_signal`, `macd_histogram`, `vroc`, `pe_ratio`, `iv`, `piotroski_score`, `fscore`, `company_info`, `bollinger_upper`, `bollinger_lower`, `bid`, `ask`, `bid_ask_spread`, `bid_ask_spread_pct`, `days_until_earnings`, `earnings_estimate_avg`, `earnings_estimate_low`, `earnings_estimate_high`, `revenue_estimate_avg`, `next_earnings_date` (string), `next_earnings_date_end` (string)
- **Timestamp**: UTC collection time (second precision)

#### `price_history` (daily prices & indicators)

- **Tags**: `ticker`
- **Fields**: `close`, `ma_50`, `ma_100`, `ma_150`, `ma_200`, `rsi`, `macd`, `macd_signal`, `macd_histogram`, `vroc`
- **Timestamp**: Trading-day date at noon UTC (second precision)

#### `price_intraday` (1-minute OHLCV bars)

- **Tags**: `ticker`
- **Fields**: `open`, `high`, `low`, `close`, `volume`
- **Timestamp**: Original minute-level timestamp from yfinance (second precision, timezone-aware)

#### `earnings_price_change` (per earnings report)

- **Tags**: `ticker`
- **Fields**: `price_before`, `price_after`, `pct_change`
- **Timestamp**: Earnings report date at noon UTC (second precision)

### Plugin Measurements (EODHD)

#### `eodhd_fundamentals` (snapshot per run)

- **Tags**: `ticker`, `exchange`, `sector`, `industry`, `type`
- **Fields**: `market_cap`, `pe_ratio`, `eps`, `dividend_yield`, `profit_margin`, `beta`, `week_52_high`, `week_52_low`, `analyst_target_price`, and many more
- **Timestamp**: UTC collection time (second precision)

#### `eodhd_financials` (quarterly/annual statements)

- **Tags**: `ticker`, `period` (`quarterly` | `annual`), `statement` (`balance_sheet` | `cash_flow` | `income_statement`)
- **Fields**: Numeric financial line items from the statement
- **Timestamp**: Filing date at 16:00 UTC (second precision)

#### `eodhd_earnings` (earnings history)

- **Tags**: `ticker`
- **Fields**: `epsActual`, `epsEstimate`, `epsDifference`, `surprisePercent`
- **Timestamp**: Earnings date at 16:00 UTC (second precision)

### Timestamp Convention

All `price_history` timestamps are set to **noon UTC (12:00:00Z)** of the trading day. This ensures dates display correctly in the Grafana dashboard regardless of the viewer's browser timezone. Midnight-UTC timestamps shift to the previous calendar day in US timezones (e.g., `2026-04-16T00:00:00Z` → April 15 8:00 PM EDT).

## Grafana Dashboard

Three dashboards are provided:

- **`grafana/dashboard.json`** — Single-ticker dashboard with stat panels, time-series charts, and technical indicators for one selected stock.
- **`grafana/dashboard-all.json`** — All-tickers dashboard with bar charts and grouped rows comparing every tracked stock side-by-side.
- **`grafana/changes.json`** — Working copy of the single-ticker dashboard with the latest panel additions and manual customizations.

An older snapshot is kept in `grafana/dashboard-modified.json`.

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

**Time-series panels** (Price History, Price & Moving Averages, RSI, MACD, VROC, Average RSI by Industry, Average IV by Industry) query `price_history`:

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

**Earnings panels** (Days Until Earnings, EPS Estimate Avg, Revenue Estimate Avg) query `stock_data`:

```flux
from(bucket: "${bucket}")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> filter(fn: (r) => r["_field"] == "days_until_earnings")
  |> group(columns: ["ticker", "_field"])
  |> last()
  |> map(fn: (r) => ({ticker: r.ticker, _value: r._value}))
  |> group()
```

In `dashboard.json` these are **stat** panels; in `dashboard-all.json` they appear inside a collapsed **Earnings Calendar** row as **bar chart** panels.

**IV vs Price** panel queries `stock_data` for both `current_price` and `iv` fields:

```flux
from(bucket: "${bucket}")
  |> range(start: v.timeRangeStart, stop: v.timeRangeStop)
  |> filter(fn: (r) => r["_measurement"] == "stock_data")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> filter(fn: (r) => r["_field"] == "current_price" or r["_field"] == "iv")
```

In `dashboard.json` / `changes.json` this is a **dual-axis timeseries** (price on left, IV % on right). In `dashboard-all.json` it is a **table** that pivots the latest values per ticker, sorted by IV descending.

**Earnings-Related Price Changes** panel queries `earnings_price_change`:

```flux
from(bucket: "${bucket}")
  |> range(start: -5y)
  |> filter(fn: (r) => r["_measurement"] == "earnings_price_change")
  |> filter(fn: (r) => r["ticker"] =~ /^${ticker:regex}$/)
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "price_before", "price_after", "pct_change"])
  |> sort(columns: ["_time"], desc: true)
```

In `dashboard.json` this is a **table** matching the earnings report view (Report Date, Price 1 Day Before, Price 1 Day After, Percentage Change). In `dashboard-all.json` it is a **horizontal bar chart** showing the most recent earnings pct_change per ticker, sorted by magnitude, with red/green color thresholds.

**Bollinger Signal** reads the tag value from the `current_price` record:

```flux
|> filter(fn: (r) => r["_field"] == "current_price")
|> group(columns: ["ticker", "_field"])
|> last()
|> map(fn: (r) => ({_value: r.bollinger_signal, ticker: r.ticker}))
|> group()
```
# Queries

## Technical Analysis

"Which stocks currently have RSI below 30 (oversold)?"
"Which stocks have RSI above 70 (overbought)?"
"Show me the MACD and MACD signal for NVDA over the last 3 months"
"Which stocks have a bullish MACD crossover (MACD above signal line)?"
"What is the current Bollinger Band signal for each stock?"
"Show me the VROC (volume rate of change) for all stocks — which have the highest volume momentum?"
"Compare the 50-day and 200-day moving averages for TSLA — is there a golden cross or death cross?"

## Fundamental Scores & Signals

"Which stocks have a Piotroski score of 8 or higher?"
"Which stocks have the highest F-score?"
"Rank all stocks by Piotroski score and show their current price"
"Compare the F-score and Piotroski score for AAPL, MSFT, and GOOGL"

## Valuation & Multiples

"Which stocks have the lowest P/E ratio?"
"Show me trailing P/E, forward P/E, P/B, P/S, and PEG ratio for all stocks in a table"
"Which stocks appear undervalued based on PEG ratio below 1?"
"Compare enterprise value to EBITDA across all stocks — which are cheapest?"
"What are the price-to-book ratios and which stocks trade below book value?"

## EODHD Fundamentals

"Which stocks have the highest analyst target price upside vs current price?"
"Show the analyst consensus (strong buy, buy, hold, sell, strong sell) for all stocks"
"Which stocks have the highest profit margin?"
"Rank stocks by return on equity (ROE)"
"Which stocks have the highest dividend yield?"
"Show me quarterly earnings growth year-over-year for all stocks"
"Which stocks have the highest quarterly revenue growth?"
"What is the beta for each stock? Which are most volatile relative to the market?"
"Which stocks have the highest short ratio or short percent of float?"

## Earnings

"Show me the EPS actual vs estimate history for NVDA — how often do they beat?"
"Which stocks had the biggest positive earnings surprise in the last quarter?"
Market Overview (cross-data)
"Show the top 10 stocks by market cap"
"How close is each stock to its 52-week high?"
"Which stocks are trading near their 52-week low?"
"What is the average RSI by industry?"
"What is the average implied volatility by industry — which sectors are most volatile?"

## Multi-factor Screening

"Find stocks with Piotroski score >= 7, RSI < 50, and positive quarterly earnings growth"
"Which stocks have analyst target upside > 20%, profit margin > 15%, and PEG < 2?"
"Show me stocks with F-score >= 7 that are also trading below their 200-day MA"
