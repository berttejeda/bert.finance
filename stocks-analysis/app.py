from datetime import datetime
from btconfig import Config
from flask import Flask, render_template, request, send_file, session
from werkzeug.exceptions import InternalServerError
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import io
import base64
import argparse
import logging
import time
import yfinance as yf
import pandas as pd
import uuid

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('stock-analyzer.main')

def parse_args():
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('--no-use-cache-first', '-no-cache', action='store_true', default=False, help='Use cached data first')
    parser.add_argument('--serve-app', '-s', action='store_true', default=True, help='Serve the data via python Flask')
    parser.add_argument('--cache-file-path', '-f', default='data.csv', help='Path to the cache file (csv)')
    parser.add_argument('--cache-expiry-in-minutes', '-c', default=5, help='How long in minutes before we expire the data cache')
    parser.add_argument('--historical-period', '-p', default="1y", help='Serve the data via python Flask')
    parser.add_argument('--debug', '-D', action='store_true', required=False, default=False)
    parser.add_argument('--verbose', '-v', action='store_true', required=False, default=False)
    return parser.parse_known_args()

# Initialize App Config
config = Config(config_file_uri='config.yaml').read()
ticker_data = config.get('tickers')

# CLI Args
args, unknown = parse_args()
debug = args.debug
no_use_cache_first = args.no_use_cache_first
cache_expiry_in_minutes = int(args.cache_expiry_in_minutes)
serve_app = args.serve_app
historical_period = args.historical_period
cache_file_obj = Path(args.cache_file_path).resolve()

# Fetch stock data
def fetch_stock_data(tickers):
    data = []
    for ticker in tickers:
        ticker_name = ticker['name']
        ticker_type = ticker['type']
        # Log messages at different levels
        logger.info(f'Retrieving price data for {ticker_name}')
        stock = yf.Ticker(ticker_name)
        pe_ratio = stock.info.get('trailingPE', None)
        hist = stock.history(period=historical_period)  # Fetch historical data

        if not hist.empty:
            current_price = round(hist["Close"].iloc[-1], 2)
            ma_50 = round(hist["Close"].rolling(window=50).mean().iloc[-1], 2)
            ma_100 = round(hist["Close"].rolling(window=100).mean().iloc[-1], 2)
            ma_200 = round(hist["Close"].rolling(window=200).mean().iloc[-1], 2)

            # Add technical indicators
            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_chart = fig_to_base64(plot_rsi(rsi))
            rsi_chart_description = """
RSI – Relative Strength Index

This is a technical indicator in stock and crypto trading.

What it measures:
- RSI measures the speed and change of price movements, showing whether a stock is overbought or oversold.
- Scale: 0 to 100
- Above 70 = Overbought (might be due for a pullback)
- Below 30 = Oversold (might be due for a rebound)

How it's used:
- Traders use RSI to spot potential reversal points.
  For example, if a stock’s RSI crosses below 30 and starts to climb back up, it might be a buy signal.
  Conversely, if RSI goes above 70 and starts to dip, that could be a sell signal.
- Example:
  A stock’s RSI is 25 → this could signal it's oversold → potential buying opportunity (after confirming with other data).
"""
            exp1 = hist['Close'].ewm(span=12, adjust=False).mean()
            exp2 = hist['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            macd_chart = fig_to_base64(plot_macd(macd))
            macd_chart_description = """
MACD – Moving Average Convergence Divergence

This is a technical indicator in stock and crypto trading.

What it measures:
- MACD tracks the relationship between two moving averages of a stock's price:
- 12-day EMA (fast)
- 26-day EMA (slow)

The MACD line is calculated as:
- MACD = 12-day EMA – 26-day EMA
- A signal line (usually a 9-day EMA of the MACD) is also plotted.
- Key elements:
  - MACD Line
  - Signal Line

Histogram – shows the difference between MACD and the Signal Line

How it’s used:
  - Bullish crossover: MACD crosses above the signal line → potential buy signal
  - Bearish crossover: MACD crosses below the signal line → potential sell signal

When the histogram bars grow/shrink, it shows increasing/decreasing momentum.

Example:
  - MACD crosses above the signal line and the histogram turns positive = bullish signal.
"""
            if ticker_type == 'stock':
                if len(stock.calendar) > 0:
                    earnings_dates = stock.calendar.get('Earnings Date', ["N/A"])
                else:
                    earnings_dates = ["N/A"]
            else:
                earnings_dates = ["N/A"]
        else:
            current_price = ma_50 = ma_100 = ma_200 = earnings_date = "N/A"
        next_earnings_date = earnings_dates[0]
        # predicted_price_movement = predict_bollinger_movement(ticker_name)
        stock_signals = bollinger_signal(ticker_name)
        stock_signal = stock_signals['Signal']
        company_info = f"""
### Company Info
- **Summary**: {stock.info.get('longBusinessSummary', 'No summary available.')}
- **Industry**: {stock.info.get('industry', 'n/a')}
- **Sector**: {stock.info.get('sector', 'n/a')}
- **Website**: [{stock.info.get('website', '#')}]({stock.info.get('website', '#')})
"""
        data.append([
            ticker_name,
            company_info,
            current_price,
            ma_50,
            ma_100,
            ma_200,
            rsi_chart,
            rsi_chart_description,
            macd_chart,
            macd_chart_description,
            stock_signal,
            pe_ratio,
            next_earnings_date
        ])
    return data

def bollinger_signal(ticker, period=25, std_dev=2):
    """
    Returns a trading signal based on Bollinger Bands:
    - 'Buy' if price < lower band
    - 'Sell' if price > upper band
    - 'Hold' if in between
    """
    df = yf.download(ticker, period='3mo', interval='1d', progress=False, auto_adjust=True)

    if df.empty or 'Close' not in df.columns:
        return {"Signal": "no data"}

    # Calculate Bollinger Bands
    df['MA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['MA'] + (std_dev * df['STD'])
    df['Lower'] = df['MA'] - (std_dev * df['STD'])
    df.dropna(inplace=True)

    if df.empty:
        return {"Signal": "no signal"}

    latest = df.iloc[-1]

    close = latest['Close'].item()
    upper = latest['Upper'].item()
    lower = latest['Lower'].item()

    if close < lower:
        signal = "Buy"
    elif close > upper:
        signal = "Sell"
    else:
        signal = "Hold"

    return {
        "Signal": signal,
        "Close": round(close, 2),
        "Upper Band": round(upper, 2),
        "Lower Band": round(lower, 2)
    }

def predict_bollinger_movement(ticker, period=20, std_dev=2):
    """
    Predicts price movement direction based on Bollinger Bands.

    Args:
        ticker (str): The stock ticker symbol.
        period (int): The moving average period.
        std_dev (int or float): Number of standard deviations for bands.

    Returns:
        str: Prediction - 'up', 'down', or 'neutral'
    """
    # Download historical data
    df = yf.download(ticker, period=historical_period, interval='1d')

    if df.empty or 'Close' not in df.columns:
        return "No data available or invalid ticker."

    # Calculate moving average and standard deviation
    df['MA'] = df['Close'].rolling(window=period).mean()
    df['STD'] = df['Close'].rolling(window=period).std()
    df['Upper'] = df['MA'] + (std_dev * df['STD'])
    df['Lower'] = df['MA'] - (std_dev * df['STD'])

    # Drop rows with NaN values
    df = df.dropna()

    # Get the last row safely
    latest_row = df.iloc[-1]

    close_price = latest_row['Close'].item()
    upper_band = latest_row['Upper'].item()
    lower_band = latest_row['Lower'].item()

    # Optional: Debug print
    print(f"Ticker: {ticker}")
    print(f"Close: {close_price}, Upper: {upper_band}, Lower: {lower_band}")

    # Decision logic
    if close_price > upper_band:
        return "down"
    elif close_price < lower_band:
        return "up"
    else:
        return "neutral"

def check_if_expired(num_minutes: int, file_path: Path) -> bool:
    """Checks if a file is older than 5 minutes.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file is older than 5 minutes, False otherwise.
    """
    if not file_path.exists() or not file_path.is_file():
        return False

    file_modification_time = file_path.stat().st_mtime
    current_time = time.time()
    difference_in_seconds = int(current_time - file_modification_time)
    is_expired = difference_in_seconds > (num_minutes * 60)
    return file_modification_time, is_expired

def compute_analysis(**kwargs):
    columns = [
        "Ticker",
        "INFO",
        "Current Price",
        "50-Day Moving Average",
        "100-Day Moving Average",
        "200-Day Moving Average",
        "RSI Chart",
        "RSI Chart Description",
        "MACD Chart",
        "MACD Chart Description",
        "Bollinger Band Signal",
        "P/E Ratio",
        "Next Earnings Date"
    ]
    session_id = kwargs['session_id']
    cache_file = cache_file_obj
    session_cache_file_obj = cache_file.parent / f'{cache_file.stem}-{session_id}{cache_file.suffix}'
    session_cache_file_path = session_cache_file_obj.as_posix()
    session_cache_file_exists = session_cache_file_obj.exists()
    cache_has_expired = False
    if session_cache_file_exists:
        file_modification_time, cache_has_expired = check_if_expired(cache_expiry_in_minutes, session_cache_file_obj)
        date_of_analysis = datetime.fromtimestamp(file_modification_time).strftime("%Y-%m-%d %H:%M:%S")
    else:
        date_of_analysis = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    refresh_cache_file = True if any([cache_has_expired, not session_cache_file_exists]) else False
    if no_use_cache_first or refresh_cache_file:
        logger.info(f'Refreshing data cache')
        stock_data = fetch_stock_data(ticker_data)
        # Create DataFrame
        df = pd.DataFrame(stock_data, columns=columns)
        df.to_csv(session_cache_file_path, index=False)
    elif session_cache_file_exists:
        logger.info(f'Using cached data from ${session_cache_file_path}')
        df = pd.read_csv(session_cache_file_path)
    else:
        raise InternalServerError('Instructed to use cache first, but no cache file found')
    return date_of_analysis, df.to_dict(orient='records')

def fetch_ticker_data(ticker, target_day):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=5)
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = pd.to_datetime(data.index)
    data['Weekday'] = data.index.weekday

    # Add technical indicators
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

    # Volume is already included

    # P/E Ratio
    pe_ratio = None
    try:
        ticker_info = yf.Ticker(ticker).info
        pe_ratio = ticker_info.get('trailingPE', None)
    except:
        pass

    # Friday and Target Day logic
    fridays = data[data['Weekday'] == 4]
    targets = data[data['Weekday'] == target_day] if isinstance(target_day, int) else data[data['Weekday'] == 4]

    results = []
    for friday_date, friday_row in fridays.iterrows():
        if target_day == "next_friday":
            target_date = friday_date + pd.Timedelta(days=7)
        else:
            days_until_target = (target_day - 4 + 7) % 7
            target_date = friday_date + pd.Timedelta(days=days_until_target)

        if target_date in targets.index:
            target_close = float(targets.loc[target_date]['Close'])
            friday_close = float(friday_row['Close'])
            change_pct = ((target_close - friday_close) / friday_close) * 100

            results.append({
                'Friday': friday_date.date(),
                'Friday Close': round(friday_close, 2),
                'Target Day': target_date.date(),
                'Target Close': round(target_close, 2),
                'Change (%)': round(change_pct, 2)
            })

    summary_df = pd.DataFrame(results)
    return data, summary_df, pe_ratio


def plot_price_chart(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['MA20'], label='MA20')
    ax.plot(data.index, data['MA50'], label='MA50')
    ax.set_title("Price and Moving Averages")
    ax.legend()
    return fig


def plot_rsi(data):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(data.index, data, label='RSI', color='purple')
    ax.axhline(70, linestyle='--', color='red')
    ax.axhline(30, linestyle='--', color='green')
    ax.set_title("RSI Chart")
    return fig


def plot_macd(data):
    signal = data.ewm(span=9, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(data.index, data, label='MACD', color='blue')
    ax.plot(data.index, signal, label='Signal', color='orange')
    ax.set_title("MACD Chart")
    ax.legend()
    return fig


def fig_to_base64(fig):
    img = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf8')

app = Flask(__name__)
app.secret_key = '123rioriu*()d-_d9432oijdcmnmcs2D'
matplotlib.use('Agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    context = {}
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())  # Generate a session ID
    session_id = session['session_id']
    if request.method == 'POST':
        # ticker = request.form['ticker']
        # target_day = request.form['target_day']
        # target_day = int(target_day) if target_day.isdigit() else target_day
        # data, summary_df, pe_ratio = fetch_ticker_data(ticker, target_day)
        date_of_analysis, analysis = compute_analysis(session_id=session_id)
        # chart = fig_to_base64(plot_price_chart(data))
        # rsi_chart = fig_to_base64(plot_rsi(data))
        # macd_chart = fig_to_base64(plot_macd(data))
        context.update({
            # 'ticker': ticker,
            # 'summary_table': summary_df.to_html(classes='table table-striped', index=False),
            # 'pe_ratio': pe_ratio,
            # 'chart': chart,
            # 'rsi_chart': rsi_chart,
            # 'macd_chart': macd_chart,
            'timestamp': date_of_analysis,
            'analysis': analysis
        })
    return render_template('index.html', **context)


@app.route('/export', methods=['POST'])
def export():
    ticker = request.form['ticker']
    target_day = request.form['target_day']
    target_day = int(target_day) if target_day.isdigit() else target_day

    _, summary_df, _ = fetch_ticker_data(ticker, target_day)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, index=False, sheet_name='Summary')
    output.seek(0)
    return send_file(output, as_attachment=True, download_name=f'{ticker}_summary.xlsx')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
