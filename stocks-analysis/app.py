from datetime import datetime, timedelta
from btconfig import Config
from fake_useragent import UserAgent
from flask import Flask, render_template, request, session
from multiprocessing import cpu_count, current_process, Pool
from pathlib import Path
from werkzeug.exceptions import InternalServerError
from finvizfinance.quote import finvizfinance
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib
import matplotlib.pyplot as plt
import nltk
import io
import base64
import argparse
import pickle
import logging
import numpy as np
import re
import requests

import time
import yfinance as yf
import pandas as pd
import tzlocal
import uuid

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('stock-analyzer.main')
# Get the local timezone
local_tz = tzlocal.get_localzone()

logger.info("Initializing NLTK Sentiment Analyzer")
# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon')
# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def parse_args():
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('--no-use-cache-first', '-no-cache', action='store_true', default=False, help='Use cached data first')
    parser.add_argument('--cache-file-path', '-f', default='data.pkl', help='Path to the cache file (pkl)')
    parser.add_argument('--cache-expiry-in-minutes', '-c', default=5, help='How long in minutes before we expire the data cache')
    parser.add_argument('--host-address', '-l', default="0.0.0.0", help='Specify host listening address')
    parser.add_argument('--threads', '-t', default=1, help='Specify max number of CPU threads for parallel processing')
    parser.add_argument('--host-port', '-p', default=5000, help='Specify host listening port')
    parser.add_argument('--historical-period', '-P', default="1y", help='Historical period')
    parser.add_argument('--debug', '-D', action='store_true', required=False, default=False)
    parser.add_argument('--verbose', '-v', action='store_true', required=False, default=False)
    return parser.parse_known_args()

# Initialize App Config
config = Config(config_file_uri='config.yaml').read()
tickers = config.get('tickers')
schema = config.get('data.schema')

# CLI Args
args, unknown = parse_args()
debug = args.debug
no_use_cache_first = args.no_use_cache_first
cache_expiry_in_minutes = int(args.cache_expiry_in_minutes)
historical_period = args.historical_period
cache_file_obj = Path(args.cache_file_path).resolve()
host_address = args.host_address
host_port = args.host_port
today = datetime.now()

try:
    num_threads = int(args.threads)
except Exception as e:
    logger.warning(f"Invalid value for number of threads {args.threads}, defaulting to 1")
    num_threads = 1
num_cpus = cpu_count()
if num_threads > num_cpus:
    num_threads = num_cpus

def refresh_yf_user_agent():
    """Refresh yfinance request session with a random User-Agent."""
    ua = UserAgent()
    session = requests.Session()
    session.headers.update({'User-Agent': ua.random})
    yf.shared._requests = session

# Fetch stock data
def fetch_stock_data(current_process_name, ticker):
    data = []
    ticker_name = ticker['name']
    ticker_type = ticker['type']
    stock = yf.Ticker(ticker_name)
    # Fetch Historical Data
    stock_history = stock.history(period=historical_period)  # Fetch historical data
    # Fetch News Headlines
    stock_news_data = fetch_ticker_news_data(ticker_name, current_process_name)
    # Fetch Additional Data
    logger.info(f'{current_process_name} - Retrieving price data for {ticker_name}')
    stock_downloaded_data = yf.download(ticker_name, period=historical_period, interval='1h', progress=False, auto_adjust=True)
    pe_ratio = stock.info.get('trailingPE', None)
    if not stock_history.empty:
        current_price = round(stock_history["Close"].iloc[-1], 2)
        ma_50 = round(stock_history["Close"].rolling(window=50).mean().iloc[-1], 2)
        ma_100 = round(stock_history["Close"].rolling(window=100).mean().iloc[-1], 2)
        ma_150 = round(stock_history["Close"].rolling(window=150).mean().iloc[-1], 2)
        ma_200 = round(stock_history["Close"].rolling(window=200).mean().iloc[-1], 2)
        # Add technical indicators
        rsi_data = calculate_rsi(ticker_name, stock_history)
        vroc_data = calculate_vroc_signals(ticker_name, stock_downloaded_data, rsi_data['RSI'])
        macd_data = calculate_macd_signals(ticker_name, stock_history)
        earnings_dates = fetch_ticker_earnings_data(ticker_type, stock)
    else:
        current_price = ma_50 = ma_100 = ma_200 = "N/A"
        earnings_dates = ["N/A"]
    next_earnings_date = earnings_dates[0]
    ma_stock_signal = get_ma_stock_signal(current_price, ma_50, ma_150)['Signal']
    bollinger_stock_data = get_bollinger_stock_data(stock_downloaded_data)
    bollinger_stock_signal = bollinger_stock_data['Signal']
    piotroski_data = calculate_piotroski_score(ticker_name, stock)
    company_data = fetch_company_info(stock)
    data.append([
        ticker_name,
        company_data['info'],
        current_price,
        company_data['industry'],
        ma_50,
        ma_100,
        ma_150,
        ma_200,
        {}, # empty dict playholder for 'Charts' column
        rsi_data['Chart'],
        rsi_data['Chart_Description'],
        macd_data['Chart'],
        macd_data['Chart_Description'],
        vroc_data['Chart'],
        vroc_data['Chart_Description'],
        stock_news_data['chart'],
        stock_news_data['chart_description'],
        stock_news_data['sentiment'],
        bollinger_stock_signal,
        ma_stock_signal,
        pe_ratio,
        piotroski_data['Score'],
        next_earnings_date
    ])
    return data

def create_news_markdown(ticker, data):
    markdown_content = f'''
<h1 style="text-align:center;">{ticker} News</h1>
<script language="javascript">
alert("ok")
</script>

{data.to_markdown()}
'''
    return markdown_content

def fetch_company_info(stock_data):
    industry = stock_data.info.get('industry', 'n/a')
    sector = stock_data.info.get('sector', 'n/a')
    website_url = stock_data.info.get('website', '#')
    website_display = website_url if website_url != '#' else 'N/A'
    company_info_markdown = f"""
### Company Info
- **Summary**: {stock_data.info.get('longBusinessSummary', 'No summary available.')}
- **Industry**: {industry}
- **Sector**: {sector}
- **Website**: [{website_display}]({website_url})
"""
    data_obj = {
        "info": company_info_markdown,
        "industry": industry,
        "sector": sector
    }
    return data_obj

def fetch_ticker_earnings_data(ticker_type, stock):
    if ticker_type == 'stock':
        if len(stock.calendar) > 0:
            earnings_dates = stock.calendar.get('Earnings Date', ["N/A"])
        else:
            earnings_dates = ["N/A"]
    else:
        earnings_dates = ["N/A"]
    return earnings_dates

def fetch_ticker_news_data(ticker, current_process_name, period=7):

    logger.info(f'{current_process_name} - Retrieving news data for {ticker}')
    stock = finvizfinance(ticker)
    news = stock.ticker_news()
    time_delta = today - timedelta(days=period)
    cutoff_date = pd.to_datetime(time_delta)
    # Convert Date column to date and filter news items
    news['Date'] = pd.to_datetime(news['Date'], unit='s', errors='coerce')
    filtered_news = news[news['Date'] > cutoff_date].reset_index(drop=True)

    filtered_news['sentiment_scores'] = filtered_news['Title'].apply(lambda content: analyzer.polarity_scores(content))
    filtered_news['compound'] = filtered_news['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])
    filtered_news['Title Sentiment'] = filtered_news['compound'].apply(
        lambda c: 'positive' if c >= 0.05 else ('negative' if c <= -0.05 else 'neutral'))
    news_sentiment = filtered_news['Title Sentiment'].mode().iloc[0]
    filtered_news = filtered_news.drop(columns=['compound', 'sentiment_scores'])

    chart = fig_to_base64(plot_stock_news_data(ticker, filtered_news))
    chart_description = create_news_markdown(ticker, filtered_news)
    data_obj = {
        'chart': chart,
        'chart_description': chart_description,
        'sentiment': news_sentiment
    }
    return data_obj

def plot_stock_news_data(ticker, data):
    # Group by formatted timestamp and count occurrences
    data['Date'] = pd.to_datetime(data['Date'], unit='s', errors='coerce', format='%d/%m/%Y').dt.strftime('%Y-%m-%d')
    counts = data.groupby('Date').size()
    fig = plt.figure(figsize=(12, 2))  # Adjust figure size as needed
    plt.plot(counts.index, counts.values, marker='o', linestyle='-')  # Line plot
    # plt.fill_between(counts.index, counts.values, alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.title(f'{ticker} News Articles Over Time')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    return fig

def net_income(ticker):
    df = ticker.income_stmt
    return df.loc['Net Income'].iloc[0]

def roa(ticker):
    df = ticker.balance_sheet
    avg_assets = (df.loc['Total Assets'].iloc[0] + df.loc['Total Assets'].iloc[1]) / 2
    return round(net_income(ticker) / avg_assets, 2)

def ocf(ticker):
    df = ticker.cash_flow
    if 'Operating Cash Flow' in df.index:
        return df.loc['Operating Cash Flow'].iloc[0]
    else:
        # Calculate Operating Cash Flow using Free Cash Flow and Captial Expenditure
        # Take the absolute value for Captial Expenditure as yf returns as a negative number
        return df.loc['Free Cash Flow'].iloc[0] + abs(df.loc['Capital Expenditure']).iloc[0]

def ltdebt(ticker):
    df = ticker.balance_sheet
    return (df.loc['Long Term Debt'].iloc[1] - df.loc['Long Term Debt'].iloc[0])

def current_ratio(ticker):
    df = ticker.balance_sheet
    current_ratio_current = df.loc['Total Assets'].iloc[0] / df.loc['Total Liabilities Net Minority Interest'].iloc[0]
    current_ratio_prev = df.loc['Total Assets'].iloc[1] / df.loc['Total Liabilities Net Minority Interest'].iloc[1]
    return round((current_ratio_current - current_ratio_prev), 2)

def new_shares(ticker):
    df = ticker.balance_sheet
    return (df.loc['Common Stock'].iloc[1] - df.loc['Common Stock'].iloc[0])

def gross_margin(ticker):
    df = ticker.income_stmt
    gross_margin_current = df.loc['Gross Profit'].iloc[0] / df.loc['Total Revenue'].iloc[0]
    gross_margin_prev = df.loc['Gross Profit'].iloc[1] / df.loc['Total Revenue'].iloc[1]
    return (gross_margin_current - gross_margin_prev)

def asset_turnover_ratio(ticker):
    df_bs = ticker.balance_sheet
    y0, y1, y2 = df_bs.loc['Total Assets'].iloc[0], df_bs.loc['Total Assets'].iloc[1], df_bs.loc['Total Assets'].iloc[2]
    avg_asset_y0 = (y0 + y1) / 2
    avg_asset_y1 = (y1 + y2) / 2

    df_is = ticker.income_stmt
    tot_rvn_y0 = df_is.loc['Total Revenue'].iloc[0] / avg_asset_y0
    tot_rvn_y1 = df_is.loc['Total Revenue'].iloc[1] / avg_asset_y1

    return round((tot_rvn_y0 - tot_rvn_y1), 2)

# Criteria name -> method
criteria_dict = {
    'CR1': net_income,
    'CR2': roa,
    'CR3': ocf,
    'CR5': ltdebt,
    'CR6': current_ratio,
    'CR7': new_shares,
    'CR8': gross_margin,
    'CR9': asset_turnover_ratio
}
CRITERIA = 'CR1,CR2,CR3,CR4,CR5,CR6,CR7,CR8,CR9,Score'.split(',')

def calculate_piotroski_score(ticker_name, stock):
    description = """
The Piotroski score is a discrete score between zero and nine that reflects 
nine criteria used to determine the strength of a firm's financial position. 
The Piotroski score is used to determine the best value stocks, 
with nine being the best and zero being the worst.    
"""

    # Dictionary to collect 9 criteria
    ps_criteria = {
        'Symbol': [], 'Name': [], 'CR1': [], 'CR2': [], 'CR3': [], 'CR4': [], 'CR5': [],
        'CR6': [], 'CR7': [], 'CR8': [], 'CR9': []
    }

    # Dictionary to collect raw data
    ps_criteria_data = {
        'Symbol': [], 'Name': [], 'CR1': [], 'CR2': [], 'CR3': [], 'CR4': [], 'CR5': [],
        'CR6': [], 'CR7': [], 'CR8': [], 'CR9': []
    }

    # Set symbol and name
    ps_criteria['Symbol'].append(stock.info['symbol'])
    ps_criteria['Name'].append(stock.info['longName'])

    ps_criteria_data['Symbol'].append(stock.info['symbol'])
    ps_criteria_data['Name'].append(stock.info['longName'])

    # 2 - Set criteria
    for key, value in criteria_dict.items():
        try:
            # 3 Uses the command pattern to call the appropriate method
            result = value(stock)
            ps_criteria_data[key].append(result)
            # 4 Special adjustment for CR7 - if there are no new shares, i.e, difference between current and previous is 0 then add 1 as well
            if key == 'CR7':
                ps_criteria[key].append(1 if result >= 0 else 0)
            else:
                # Process with other CRs
                ps_criteria[key].append(1 if result > 0 else 0)
        except (KeyError, IndexError) as err:
            # 5 Error encountered, due to missing data"
            logger.debug(f'Missing piotroski criteria for {ticker_name} - {err}')
            ps_criteria[key].append(0)
            ps_criteria_data[key].append(np.nan)

    # 6 CR4 - handle it differently as it doesn't invoke a method
    # CR4 - Cash flow from operations being greater than net income (quality of earnings)
    if ps_criteria_data['CR3'][-1] > ps_criteria_data['CR1'][-1]:
        ps_criteria['CR4'].append(1)
        ps_criteria_data['CR4'].append(1)
    else:
        # Set criteria and raw data to false (0)
        ps_criteria['CR4'].append(0)
        ps_criteria_data['CR4'].append(0)

    ps_criteria_df = pd.DataFrame(ps_criteria)
    # Add ranking scores to get the total score
    ps_criteria_df['Score'] = ps_criteria_df[CRITERIA[:-1]].sum(axis=1).iloc[-1]
    data_obj = {
        "Description": description,
        "Score": ps_criteria_df['Score'][0]
    }
    return data_obj

def calculate_rsi(ticker, data, period=7):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi_chart = fig_to_base64(plot_rsi(ticker, rsi))
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

    data_obj = {
        'Chart': rsi_chart,
        'Chart_Description': rsi_chart_description,
        'RSI': rsi
    }

    return data_obj

def calculate_vroc_signals(ticker, data, rsi, period=7, vroc_buy_threshold=20, vroc_sell_threshold=-20, rsi_buy_threshold=30, rsi_sell_threshold=70):
    # Function to calculate VROC
    data['VROC'] = data['Volume'].pct_change(periods=period) * 100
    # Signal generation
    data['Signal'] = 0
    data.loc[(data['VROC'] > vroc_buy_threshold) & (rsi < rsi_buy_threshold), 'Signal'] = 1
    data.loc[(data['VROC'] < vroc_sell_threshold) & (rsi > rsi_sell_threshold), 'Signal'] = -1
    vroc_signal_chart_description = """
VROC - Volume Rate of Change

This indicator measures the percentage change in trading volume over a specific period. 
A rising VROC could signal increasing interest in a stock.
"""
    data_obj = {
        'Chart': fig_to_base64(plot_vroc(ticker, data)),
        'Chart_Description': vroc_signal_chart_description
    }
    return data_obj

def plot_vroc(ticker, data):
    fig = plt.figure(figsize=(12, 2))
    plt.title(f'{ticker} Volume Rate of Change (VROC) Chart')
    plt.plot(data['VROC'], label='VROC', color='orange')
    return fig

def get_ma_stock_signal(current_price, ma_50, ma_150):

    if current_price > ma_50 and current_price > ma_150:
        ma_signal = "Buy"
    elif current_price < ma_50 and current_price < ma_150:
        ma_signal = "Sell"
    else:
        ma_signal = "Hold"
    data_obj = {
        "Signal": ma_signal,
    }
    return data_obj


def get_bollinger_stock_data(data, period=20, std_dev=2):
    """
    Returns trading signals based on various factors, e.g. Bollinger Bands:
    - MovingAverageSignal:
        - 'Buy' if price > 50-Day Moving Average && price > 150-Day Moving Average
        - 'Sell' if price < 50-Day Moving Average && price < 150-Day Moving Average
        - else, 'Hold'
    - Bollinger:
        - 'Buy' if price < lower band
        - 'Sell' if price > upper band
        - 'Hold' if in between
    """
    if data.empty or 'Close' not in data.columns:
        return {"Signal": "no data"}

    time_delta = today - timedelta(days=period)
    cutoff_date = pd.to_datetime(time_delta)
    # Convert Date column to date and filter news items
    filtered_data = data[data.index < str(cutoff_date)]
    data.update(filtered_data)
    # Calculate Bollinger Bands
    data['MA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper'] = data['MA'] + (std_dev * data['STD'])
    data['Lower'] = data['MA'] - (std_dev * data['STD'])
    data.dropna(inplace=True)
    latest = data.iloc[-1]
    close = latest['Close'].item()
    upper = latest['Upper'].item()
    lower = latest['Lower'].item()

    if close < lower:
        bollinger_signal = "Buy"
    elif close > upper:
        bollinger_signal = "Sell"
    else:
        bollinger_signal = "Hold"

    data_obj = {
        "Signal": bollinger_signal,
        "Close": round(close, 2),
        "Upper Band": round(upper, 2),
        "Lower Band": round(lower, 2)
    }

    return data_obj

def check_if_expired(num_minutes: int, file_path: Path) -> [datetime, bool]:
    """Checks if a file is older than num_minutes.

    Args:
        file_path: The path to the file.

    Returns:
        True if the file is older than 5 minutes, False otherwise.
    """
    if not file_path.exists() or not file_path.is_file():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return timestamp, True
    else:
        file_modification_time = file_path.stat().st_mtime
        current_time = time.time()
        difference_in_seconds = int(current_time - file_modification_time)
        is_expired = difference_in_seconds > (num_minutes * 60)
        timestamp = datetime.fromtimestamp(file_modification_time).strftime("%Y-%m-%d %H:%M:%S")
        return timestamp, is_expired

def compute_analysis(ticker):
    current_process_name = current_process().name
    stock_data = fetch_stock_data(current_process_name, ticker)
    # Create DataFrame
    df = pd.DataFrame(stock_data, columns=schema)
    return df.to_dict(orient='records')

def plot_price_chart(data):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close')
    ax.plot(data.index, data['MA20'], label='MA20')
    ax.plot(data.index, data['MA50'], label='MA50')
    ax.set_title("Price and Moving Averages")
    ax.legend()
    return fig

def plot_rsi(ticker, data):
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(data.index, data, label='RSI', color='purple')
    ax.axhline(70, linestyle='--', color='red')
    ax.axhline(30, linestyle='--', color='green')
    ax.set_title(f'{ticker} RSI Chart')
    ax.legend()
    return fig

def calculate_macd_signals(ticker, data):
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    macd_chart = fig_to_base64(plot_macd(ticker, macd))
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
    data_obj = {
        'Chart': macd_chart,
        'Chart_Description': macd_chart_description
    }
    return data_obj

def plot_macd(ticker, data):
    signal = data.ewm(span=9, adjust=False).mean()
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.plot(data.index, data, label='MACD', color='blue')
    ax.plot(data.index, signal, label='Signal', color='orange')
    ax.set_title(f'{ticker} MACD Chart')
    ax.legend()
    return fig

def fig_to_base64(fig):
    img = io.BytesIO()
    fig.tight_layout()
    fig.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)
    return base64.b64encode(img.read()).decode('utf8')

# Custom jinja regex filter
def regex_search(value, pattern):
    match = re.search(pattern, value)
    return match.group(0) if match else ''

app = Flask(__name__)
app.secret_key = '123rioriu*()d-_d9432oijdcmnmcs2D'
app.jinja_env.filters['regex_search'] = regex_search
matplotlib.use('Agg')

@app.route('/', methods=['GET', 'POST'])
def index():
    refresh_yf_user_agent()
    context = {}
    if request.method == 'POST':
        logger.info("Main process - Starting stock data fetch")
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())  # Generate a session ID
        session_id = session['session_id']
        start_time = time.time()
        session_cache_file_obj = cache_file_obj.parent / f'{cache_file_obj.stem}-{session_id}{cache_file_obj.suffix}'
        session_cache_file_path = session_cache_file_obj.as_posix()
        session_cache_file_exists = session_cache_file_obj.exists()
        date_of_analysis, cache_has_expired = check_if_expired(cache_expiry_in_minutes, session_cache_file_obj)
        if no_use_cache_first or cache_has_expired:
            logger.info(f'Refreshing data cache')
            with Pool(processes=num_threads) as pool:
                stock_data_analysis = pool.map(compute_analysis, tickers)
                with open(session_cache_file_path, 'wb') as f:
                    pickle.dump(stock_data_analysis, f)
        elif session_cache_file_exists:
            logger.info(f'Using cached data from {session_cache_file_path}')
            with open(session_cache_file_path, 'rb') as f:
                stock_data_analysis = pickle.load(f)
        else:
            raise InternalServerError('Instructed to use cache first, but no cache file found')
        context.update({
            'timestamp': date_of_analysis,
            'stock_data_analysis': stock_data_analysis
        })
        duration = time.time() - start_time
        logger.info(f"Main process - Completed stock data fetch in {duration:.2f} seconds")
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(host=host_address, port=host_port, debug=debug)
