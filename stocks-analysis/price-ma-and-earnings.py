import argparse
import logging
import time

import yfinance as yf
import pandas as pd

from btconfig import Config
from flask import Flask, render_template
from pathlib import Path


# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('stock-analyzer.main')

def parse_args():
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('--no-use-cache-first', '-no-cache', action='store_true', default=False, help='Use cached data first')
    parser.add_argument('--serve-app', '-s', action='store_true', default=True, help='Serve the data via python Flask')
    parser.add_argument('--cache-expiry-in-minutes', '-c', default=5, help='How long in minutes before we expire the data cache')
    parser.add_argument('--historical-period', '-p', default="6mo", help='Serve the data via python Flask')
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

# Create an empty DataFrame

# Fetch stock data
def fetch_stock_data(tickers):
    data = []
    for ticker in tickers:
        ticker_name = ticker['name']
        ticker_type = ticker['type']
        # Log messages at different levels
        logger.info(f'Retrieving price data for {ticker_name}')
        stock = yf.Ticker(ticker_name)
        hist = stock.history(period=historical_period)  # Fetch historical data

        if not hist.empty:
            current_price = hist["Close"].iloc[-1]
            ma_50 = hist["Close"].rolling(window=50).mean().iloc[-1]
            ma_100 = hist["Close"].rolling(window=100).mean().iloc[-1]
            if ticker_type == 'stock':
                if len(stock.calendar) > 0:
                    earnings_dates = stock.calendar.get('Earnings Date', ["N/A"])
                else:
                    earnings_dates = ["N/A"]
            else:
                earnings_dates = ["N/A"]
        else:
            current_price = ma_50 = ma_100 = earnings_date = "N/A"
        next_earnings_date = earnings_dates[0]
        # predicted_price_movement = predict_bollinger_movement(ticker_name)
        stock_signals = bollinger_signal(ticker_name)
        stock_signal = stock_signals['Signal']
        data.append([ticker_name, current_price, ma_50, ma_100, stock_signal, next_earnings_date])
    return data

def bollinger_signal(ticker, period=25, std_dev=2):
    """
    Returns a trading signal based on Bollinger Bands:
    - 'Buy' if price < lower band
    - 'Sell' if price > upper band
    - 'Hold' if in between
    """
    df = yf.download(ticker, period='3mo', interval='1d', progress=False)

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

    return difference_in_seconds > (num_minutes * 60)

app = Flask(__name__, template_folder=Path(".").resolve())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/update_content', methods=['POST'])
async def update_content():
    columns = ["Ticker", "Current Price", "50-Day MA", "100-Day MA", "Stock Signal", "Next Earnings Date"]
    cache_file = Path('price-ma-and-earnings.csv')
    cache_file_exists = cache_file.exists()
    cache_has_expired = False
    if cache_file_exists:
        cache_has_expired = check_if_expired(cache_expiry_in_minutes, cache_file)
    refresh_cache_file = True if any([cache_has_expired, not cache_file_exists]) else False
    if no_use_cache_first or refresh_cache_file:
        logger.info(f'Refreshing data cache')
        stock_data = fetch_stock_data(ticker_data)
        # Create DataFrame
        df = pd.DataFrame(stock_data, columns=columns)
        df.to_csv('price-ma-and-earnings.csv', index=False)
    elif Path('price-ma-and-earnings.csv').exists():
        logger.info(f'Using cached data from price-ma-and-earnings.csv')
        df = pd.read_csv('price-ma-and-earnings.csv')
    else:
        quit('Instructed to use cache first, but no cache file found')
    return render_template('partial.html', data=df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(port=8000, host="0.0.0.0", debug=debug)
