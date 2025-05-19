from btconfig import Config
from datetime import datetime, timedelta
from finance_calendars import finance_calendars as fc
from flask import Flask, render_template
from tasks import fetch_stock_data_parallel, get_cache_key, cache
from io import StringIO

import argparse
import glob
import json
import logging
import pandas as pd
import re
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Stock Analysis Script")
    parser.add_argument('--no-use-cache-first', '-no-cache', action='store_true', default=False, help='Use cached data first')
    parser.add_argument('--host-address', '-l', default="0.0.0.0", help='Specify host listening address')
    parser.add_argument('--host-port', '-p', default=8080, help='Specify host listening port')
    # parser.add_argument('--finnhub-api-key', '-fhk', default=os.environ.get('FINNHUB_API_KEY'), help='Specify Finnhub API Key')
    parser.add_argument('--historical-period', '-P', default="1y", help='Historical period')
    parser.add_argument('--debug', '-D', action='store_true', required=False, default=False)
    parser.add_argument('--verbose', '-v', action='store_true', required=False, default=False)
    return parser.parse_known_args()

# CLI Args
args, unknown = parse_args()
debug = args.debug
no_use_cache_first = args.no_use_cache_first
historical_period = args.historical_period
host_address = args.host_address
host_port = args.host_port
today = datetime.now()
# finnhub_api_key = args.finnhub_api_key or quit('You must specify your Finnhub API Key!')
# Initialize App Config
config = Config(config_file_uri='config.yaml').read()
tickers = [t['name'] for t in config.get('tickers') if t['type'] == 'stock']
schema = config.get('data.schema')
# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('stock-analyzer.app')
# Initialize Flask App
app = Flask(__name__)
# Custom jinja regex filter
def regex_search(value, pattern):
    match = re.search(pattern, value)
    return match.group(0) if match else ''
app.jinja_env.filters['regex_search'] = regex_search

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

def earnings_calendar(ticker_db):
    monday = today - timedelta(days=today.weekday())
    weekdays = [monday + timedelta(days=i) for i in range(5)]

    logger.info('Fetching earnings calendar for the week')
    earnings_by_day = {}
    ticker_list = []
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
                ticker_list =  ticker_list + [t for t in df['ticker']]
                rows = df.to_dict(orient='records')
            else:
                rows = []
        except Exception:
            rows = []
        earnings_by_day[day.strftime('%A, %Y-%m-%d')] = rows

    return ticker_list, earnings_by_day
ticker_db = build_ticker_db()

# Gather earnings data for the week
ticker_list_from_earnings, earnings_by_day = earnings_calendar(ticker_db)
combined_ticker_list = ticker_list_from_earnings + tickers
# List to store processed ticker objects
sanitized_tickers = []
for t in combined_ticker_list:
    if t != None and t not in sanitized_tickers:
        sanitized_tickers.append(t)
# Initialize the stock data cache
logger.info('Initializing the stock data cache ...')
start_time = time.time()
chord_result = fetch_stock_data_parallel(sanitized_tickers)  # returns AsyncResult
if type(chord_result) != str:
    chord_result.wait()
duration = time.time() - start_time
logger.info(f"Completed initial stock data fetch in {duration:.2f} seconds")
initial_refresh_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@app.route('/')
def home():
    date_of_analysis = initial_refresh_date
    context = {
        'stock_data_is_ready': False,
        'earnings_by_day': earnings_by_day,
        'date_of_analysis':  date_of_analysis
    }
    key = get_cache_key(tickers)
    cached = cache.get(key)
    if cached:
        logger.info('Loading cached data')
        stock_data_analysis = pd.read_json(StringIO(cached.decode('utf-8')))
    else:
        start_time = time.time()
        chord_result = fetch_stock_data_parallel(sanitized_tickers)  # returns AsyncResult
        if type(chord_result) == str:
            stock_data_analysis = chord_result
        else:
            chord_result.wait()
            stock_data_analysis = pd.read_json(StringIO(chord_result.get()))
            duration = time.time() - start_time
            logger.info(f"Completed stock data fetch in {duration:.2f} seconds")
            date_of_analysis = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    context.update(
      {
      'timestamp': date_of_analysis,
      'stock_data_is_ready': True,
      'stock_data_analysis': stock_data_analysis.to_dict(orient='records')
      }
    )    
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(host=host_address, port=host_port, debug=debug)