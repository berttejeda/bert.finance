import redis
import hashlib
from celery import Celery, chord
from datetime import datetime, timedelta
from finvizfinance.quote import finvizfinance
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import base64
import io
import logging
import matplotlib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import time
import yfinance as yf

cache = redis.Redis(host='localhost', port=6379, db=1)

CACHE_TTL = 300

celery = Celery('tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'  # NEW: result backend
)
celery.conf.event_serializer = 'pickle'
celery.conf.task_serializer = 'pickle'
celery.conf.result_serializer = 'pickle'
celery.conf.accept_content = ['application/json', 'application/x-python-serialize']

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('celery-task')

logger.info("Initializing NLTK Sentiment Analyzer")
# Ensure necessary NLTK data is downloaded
nltk.download('vader_lexicon')
# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
today = datetime.now()
matplotlib.use('Agg')
mlogger = logging.getLogger('matplotlib')
mlogger.setLevel(logging.WARNING)
historical_period = '1y'

def get_cache_key(tickers):
    joined = ",".join(sorted(tickers))
    return f"stock:{hashlib.md5(joined.encode()).hexdigest()}"

@celery.task
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        # Fetch Historical Data
        stock_history = stock.history(period=historical_period)  # Fetch historical data
        # Fetch News Headlines
        stock_news_data = fetch_ticker_news_data(ticker)
        # Fetch Additional Data
        logger.info(f'Retrieving price data for {ticker}')
        stock_downloaded_data = yf.download(ticker, period=historical_period, interval='1h', progress=False,
                                            auto_adjust=True)
        if not stock_history.empty:
            # current_price = round(stock_history["Close"].iloc[-1], 2)
            current_price = stock.info.get('regularMarketPrice')
            ma_50 = round(stock_history["Close"].rolling(window=50).mean().iloc[-1], 2)
            ma_100 = round(stock_history["Close"].rolling(window=100).mean().iloc[-1], 2)
            ma_150 = round(stock_history["Close"].rolling(window=150).mean().iloc[-1], 2)
            ma_200 = round(stock_history["Close"].rolling(window=200).mean().iloc[-1], 2)
            # Add technical indicators
            rsi_data = calculate_rsi(ticker, stock_history)
            vroc_data = calculate_vroc_signals(ticker, stock_downloaded_data, rsi_data['RSI'])
            macd_data = calculate_macd_signals(ticker, stock_history)
        else:
            current_price = ma_50 = ma_100 = ma_200 = "N/A"
            earnings_dates = ["N/A"]
        ma_stock_signal = get_ma_stock_signal(current_price, ma_50, ma_150)['Signal']
        bollinger_stock_data = get_bollinger_stock_data(stock_downloaded_data)
        bollinger_stock_signal = bollinger_stock_data['Signal']
        try:
            piotroski_data = calculate_piotroski_score(ticker, stock)
        except Exception as e:
            logger.warn(f'Could not determine Piotroski Score for {ticker}, error was {e}')
            piotroski_data = {'Score': ''}
        company_data = fetch_company_info(stock)
        try:
            earnings_date_timestamps = stock.calendar.get('Earnings Date')
            first_earnings_date_timestamp = earnings_date_timestamps[0]
            if first_earnings_date_timestamp < today.date():
                next_earnings_date_timestamp = earnings_date_timestamps[-1]
            else:
                next_earnings_date_timestamp = first_earnings_date_timestamp
            logger.info(f'Next earnings date for {ticker} is {next_earnings_date_timestamp}')
            next_earnings_date = next_earnings_date_timestamp.strftime("%Y-%m-%d")
        except Exception as e:
            logger.error(f'Failed to retrieve next earnings date for {ticker}, error was {e}')
            next_earnings_date = 'N/A'
        data_obj = {
            'Ticker': ticker,
            'Info': company_data['info'],
            'Price': current_price,
            'Market Cap': stock.info.get('marketCap'),
            'Industry': company_data['industry'],
            '50-MA': ma_50,
            '100-MA': ma_100,
            '100-MA': ma_100,
            '150-MA': ma_150,
            '200-MA': ma_200,
            '52w High': stock.info.get('fiftyTwoWeekHigh'),
            '52w Low': stock.info.get('fiftyTwoWeekLow'),
            'Charts': {}, # empty dict playholder for 'Charts' column
            'RSI Chart': rsi_data['Chart'],
            'RSI Chart Description': rsi_data['Chart_Description'],
            'MACD Chart': macd_data['Chart'],
            'MACD Chart Description': macd_data['Chart_Description'],
            'VROC Chart': vroc_data['Chart'],
            'VROC Chart Description': vroc_data['Chart_Description'],
            'News Chart': stock_news_data['chart'],
            'News Chart Description': stock_news_data['chart_description'],
            'Sentiment': stock_news_data['sentiment'],
            'Σ-BOLL': bollinger_stock_signal,
            'Σ-MA': ma_stock_signal,
            'P/E': stock.info.get('trailingPE'),
            'Score': piotroski_data['Score'],
            'Earnings': next_earnings_date,
        }
    except Exception as e:
        data_obj = {'Ticker': ticker, 'Error': str(e)}
    return data_obj

@celery.task
def collect_results(results):
    df = pd.DataFrame(results)
    json_data = df.to_json()
    key = get_cache_key([row['Ticker'] for row in results if 'Ticker' in row])
    cache.setex(key, CACHE_TTL, json_data)
    return json_data

def fetch_stock_data_parallel(tickers):
    key = get_cache_key(tickers)
    cached = cache.get(key)
    if cached:
        return cached.decode('utf-8')
    else:
        # Return full AsyncResult
        return chord(
            [fetch_stock_data.s(ticker) for ticker in tickers]
        )(collect_results.s())

def create_news_markdown(ticker, data):
    markdown_content = f'''
<h1 style="text-align:center;">{ticker} News</h1>
<script language="javascript">
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

def fetch_ticker_news_data(ticker, period=7, retries=3):

    logger.info(f'Retrieving news data for {ticker}')
    is_error = False
    for i in range(retries):
        try:
            stock = finvizfinance(ticker)
            news = stock.ticker_news()
            time_delta = today - timedelta(days=period)
            cutoff_date = pd.to_datetime(time_delta)
            # Convert Date column to date and filter news items
            news['Date'] = pd.to_datetime(news['Date'], unit='s', errors='coerce')
            filtered_news = news[news['Date'] > cutoff_date].reset_index(drop=True)

            filtered_news['sentiment_scores'] = filtered_news['Title'].apply(
                lambda content: analyzer.polarity_scores(content))
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
            is_error = False
        except Exception as e:
            logger.error(f'Failed to retrieve news data for {ticker} (error was {e}, retrying ...')
            is_error = True
            time.sleep(5 * (i + 1)) # Exponential backoff
    if is_error:
        logger.error(f'Reached max number of retries {retries} when fetching news data for ticker {ticker}')
        data_obj = {
            'chart': fig_to_base64(plot_empty_data(ticker)),
            'chart_description': '',
            'sentiment': ''
        }
    return data_obj

def plot_empty_data(ticker):
    # Create an empty plot
    fig = plt.figure(figsize=(12, 2))
    plt.plot([], [])
    # Set labels and title
    plt.title(f"No data available for {ticker}")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    # Show the plot
    return fig

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

def calculate_piotroski_score(ticker, stock):
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
            logger.debug(f'Missing piotroski criteria for {ticker} - {err}')
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