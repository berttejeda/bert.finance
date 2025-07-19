from btconfig import Config
from celery import Celery, chord
from celery.schedules import crontab
from datetime import datetime, timedelta
from fake_useragent import UserAgent
from finance_calendars import finance_calendars as fc
from lib.api_client import PolygonTicker
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests import Session

import hashlib
import json
import glob
import logging
import matplotlib
import nltk
import os
import pandas as pd
import redis
import sys
import time
import traceback
import yfinance as yf

# Initialize App Config
config = Config(config_file_uri='config.yaml').read()
sic_data = Config(config_file_uri='etc/sic.yaml').read()
celery = Celery('tasks',
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/1')

celery.conf.beat_schedule = {
    'fetch-stock-every-30-min': {
        'task': 'tasks.fetch_and_cache_all_tickers',
        'schedule': crontab(minute='*/30'),
        'args': (1,),
    }
}

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
redis_host = 'localhost'
redis_port = 6379
redis_db = 1

# current_limit = sys.getrecursionlimit()
# logger.info(f"Current recursion limit: {current_limit}")
# # Set a new recursion limit
# new_limit = 6000
# sys.setrecursionlimit(new_limit)
# logger.info(f"Set new recursion limit: {new_limit}")

try:
    cache = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    cache.ping()
except redis.exceptions.ConnectionError:
    quit(f'Could not connect to redis at {redis_host}:{redis_port}')

CACHE_TTL_STOCKS = 432000 # cache stock data for 5 hours
CACHE_TTL_EARNINGS = 432000 # cache earnings data for 5 days
QUIVER_API_KEY = os.environ.get('QUIVER_API_KEY')  # Replace with actual key
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
end = datetime.utcnow()
start = end - timedelta(days=365)

def get_current_vix():
    '''
    Function to return the current value of VIX, the Yahoo Finance ticker for the CBOE Volatility Index
    '''
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d", interval="1m")  # get today's minute-level data
    latest_price = None
    if not vix_data.empty:
        latest_price = int(vix_data['Close'].iloc[-1])
    return latest_price

def refresh_yf_user_agent():
    """Refresh yfinance request session with a random User-Agent."""
    ua = UserAgent()
    session = Session()
    session.headers.update({'User-Agent': ua.random})
    yf.shared._requests = session

refresh_yf_user_agent()

# Build dictionary of existing US stocks
def build_ticker_db():
    # List to hold combined ticker data
    ticker_data = []
    # Iterate over all JSON files
    for file in glob.glob("etc/*.json"):
        with open(file, 'r') as f:
            data = json.load(f)  # Assuming each file contains a list of dicts
            ticker_data.extend(data)
    return ticker_data

def get_ticker_from_db(data, search_key):
    for item in data:
        company_name = item.get('name', '')
        matches_name_bare = search_key in company_name
        matches_name_adjusted = search_key.replace(',','').replace('.','') in company_name
        if matches_name_bare or matches_name_adjusted:
            return item['symbol']
    return None

def extract_key_values(data, key):
    """
    Extracts values associated with a specific key from a dictionary of lists of dictionaries.

    Args:
        data (dict): The dictionary containing a list of dictionaries.
        key (str): The key whose values need to be extracted.

    Returns:
        list: A list of values associated with the specified key.
    """
    values = []
    for item in data.values():
        if isinstance(item, list):
            for dictionary in item:
                if key in dictionary and dictionary.get(key) is not None:
                    values.append(dictionary[key])
    return values

def earnings_calendar(ticker_db):
    cache_key = "earnings_this_week"
    cached_data = cache.get(cache_key)
    ticker_list = []
    if cached_data:
        earnings_by_day = json.loads(cached_data)
        ticker_list = extract_key_values(earnings_by_day, 'ticker')
        return ticker_list, earnings_by_day
    else:
        monday = today - timedelta(days=today.weekday())
        weekdays = [monday + timedelta(days=i) for i in range(5)]
        logger.info('Fetching earnings calendar for the week')
        earnings_by_day = {}
        for day in weekdays:
            try:
                df = fc.get_earnings_by_date(day)
                df['ticker'] = df['name'].apply(lambda x: get_ticker_from_db(ticker_db, x))
                if 'eps' in df.keys():
                    df['eps'] = df['eps'].str.replace('$', '')
                if 'noOfEsts' in df.keys():
                    df['noOfEsts'] = df['noOfEsts'].str.replace('N/A', '')
                if 'lastYearEPS' in df.keys():
                    df['lastYearEPS'] = df['lastYearEPS'].str.replace('$', '').str.replace('(', '-').str.replace(')', '')
                    df['lastYearEPS'] = df['lastYearEPS'].str.replace('N/A', '')
                if 'epsForecast' in df.keys():
                    df['epsForecast'] = df['epsForecast'].str.replace('$', '').str.replace('(','-').str.replace(')','')
                if not df.empty:
                    ticker_list = ticker_list + [t for t in df['ticker']]
                    rows = df.to_dict(orient='records')
                else:
                    rows = []
            except Exception:
                rows = []
            earnings_by_day[day.strftime('%A, %Y-%m-%d')] = rows
        cache.setex(cache_key, CACHE_TTL_EARNINGS, json.dumps(earnings_by_day))
        return ticker_list, earnings_by_day
    
tickers_from_config = [t['name'] for t in config.get('tickers') if t['type'] == 'stock']
ticker_db = build_ticker_db()
# Gather earnings data for the week
ticker_list_from_earnings, earnings_by_day = earnings_calendar(ticker_db)
combined_ticker_list = ticker_list_from_earnings + tickers_from_config
# List to store processed ticker objects
tickers = []
for t in combined_ticker_list:
    if t != None and t not in tickers:
        tickers.append(t)

def get_cache_key(tickers):
    joined = ",".join(sorted(tickers))
    return f"stock:{hashlib.md5(joined.encode()).hexdigest()}"

@celery.task
def collect_results(results, fetch_stock_data_parallel=False):
    df = pd.DataFrame(results)
    json_data = df.to_json()
    list_of_tickers = [r['Ticker'] for r in results if 'Ticker' in r]
    key = get_cache_key(list_of_tickers)
    cache.setex(key, CACHE_TTL_STOCKS, json_data)
    return json_data

def fetch_stock_data_parallel(tickers, fetch_stock_data_parallel=False):
    output = chord(
        [fetch_one_ticker.s(ticker) for ticker in tickers]
    )(collect_results.s(fetch_stock_data_parallel))
    return output

@celery.task
def fetch_one_ticker(ticker):
    start_time = time.time()
    try:
        stock = PolygonTicker(symbol=ticker, polygon_api_key=POLYGON_API_KEY, quiver_api_key=QUIVER_API_KEY, sic_data=sic_data)
        current_price = stock.current_price
        ma_50 = stock.sma_50
        ma_100 = stock.sma_100
        ma_150 = stock.sma_150
        ma_200 = stock.sma_200
        price_data_last_trading_month = stock.price_data_last_trading_month
        rsi_data = stock.rsi_data
        vroc_data = stock.vroc_data
        macd_data = stock.macd_data
        stock_news_data = stock.news_data
        ma_stock_signal = stock.sma_stock_signal['Signal']
        bollinger_stock_data = stock.bollinger_stock_data
        bollinger_stock_signal = bollinger_stock_data['Signal']
        piotroski_data = stock.patrioski_data
        company_info = stock.company_info
        next_earnings_date = stock.next_earnings_date.get('date', 'N/A')
        senator_trades_data = stock.senator_trade_data
        duration = time.time() - start_time
        completed_at = datetime.utcnow().isoformat()
        market_cap = f'{stock.ticker_details_current_year.market_cap/1e9:.2f}B'
        logger.info(f"Completed stock data fetch in {duration:.2f} seconds")
        data_obj = {
            'Ticker': ticker,
            'Info': company_info['summary'],
            'Price': current_price,
            'Market Cap': market_cap,
            'Industry': company_info['industry'],
            '50-MA': ma_50,
            '100-MA': ma_100,
            '100-MA': ma_100,
            '150-MA': ma_150,
            '200-MA': ma_200,
            '52w High': stock.fifty_two_week_high,
            '52w Low': stock.fifty_two_week_low,
            'Charts': {}, # empty dict playholder for 'Charts' column
            'Price Chart': price_data_last_trading_month['Chart'],
            'Price Chart Description': price_data_last_trading_month['Chart Description'],
            'RSI Chart': rsi_data['Chart'],
            'RSI Chart Description': rsi_data['Chart Description'],
            'MACD Chart': macd_data['Chart'],
            'MACD Chart Description': macd_data['Chart Description'],
            'VROC Chart': vroc_data['Chart'],
            'VROC Chart Description': vroc_data['Chart Description'],
            'Senate Trades Chart': senator_trades_data['Chart'],
            'Senate Trades Chart Description': senator_trades_data['Chart Description'],
            'News Chart': stock_news_data['Chart'],
            'News Chart Description': stock_news_data['Chart Description'],
            'Sentiment': stock_news_data['overall_sentiment'],
            'Σ-BOLL': bollinger_stock_signal,
            'Σ-MA': ma_stock_signal,
            'P/E': stock.trailing_pe,
            'Score': piotroski_data['Score'],
            'Earnings': next_earnings_date,
            'Duration': duration,
            'CompletedAt': completed_at,
        }
    except Exception as e:
        completed_at = datetime.utcnow().isoformat()
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # Extract the file name and line number from the traceback
        line_number = traceback.extract_tb(exc_traceback)[-1][1]
        file_name = traceback.extract_tb(exc_traceback)[-1][0]
        data_obj = {'Ticker': ticker, 'Error': str(e), 'ErrorFile': file_name,'ErrorLineNumber': line_number, 'CompletedAt': completed_at}
    return data_obj

@celery.task(name='tasks.fetch_and_cache_all_tickers')
def fetch_and_cache_all_tickers(force_cache_update=0):
    if force_cache_update:
        logger.info('Got signal to force cache update')
    fetch_stock_data_parallel(tickers)

def start_worker():

    sys.argv = [
        'worker',
        '--loglevel=INFO',
        '-c 10',
    ]
    celery.worker_main(argv=sys.argv)

if __name__ == '__main__':
    fetch_and_cache_all_tickers()
    start_worker()