from btconfig import Config
from datetime import datetime, timedelta
from finance_calendars import finance_calendars as fc
from flask import Flask, render_template
from tasks import fetch_stock_data, get_cache_key, cache
import glob
import json
import logging
import pandas as pd
import re

# Initialize App Config
config = Config(config_file_uri='config.yaml').read()
tickers = [t['name'] for t in config.get('tickers') if t['type'] == 'stock']
schema = config.get('data.schema')
today = datetime.now()

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('stock-analyzer.app')

app = Flask(__name__)

# Custom jinja regex filter
def regex_search(value, pattern):
    match = re.search(pattern, value)
    return match.group(0) if match else ''

app.jinja_env.filters['regex_search'] = regex_search

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

    logger.info('Fetching earnlings calendar for the week')
    earnings_by_day = {}
    ticker_list = []
    seen = set()
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
                ticker_list =  ticker_list + [{'name': t, 'type': 'stock'} for t in df['ticker']]
                rows = df.to_dict(orient='records')
            else:
                rows = []
        except Exception:
            rows = []
        earnings_by_day[day.strftime('%A, %Y-%m-%d')] = rows

    return ticker_list, earnings_by_day

ticker_db = build_ticker_db()
ticker_list_from_earnings, earnings_by_day = earnings_calendar(ticker_db)

@app.route('/')
def home():
    context = {
        'stock_data_is_ready': False,
        'earnings_by_day': earnings_by_day
    }
    key = get_cache_key(tickers)
    cached = cache.get(key)
    if cached:
        stock_data_analysis = pd.read_json(cached.decode('utf-8'))
    else:
        result = fetch_stock_data.apply_async(args=[tickers])
        result.wait()
        stock_data_analysis = pd.read_json(result.get())

    # stock_data_analysis['Market Cap'] = stock_data_analysis['Market Cap'].apply(lambda x: f"${x/1e9:.2f}B" if pd.notnull(x) else x)
    # stock_data_analysis['Price'] = stock_data_analysis['Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else x)
    # stock_data_analysis['52w High'] = stock_data_analysis['52w High'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else x)
    # stock_data_analysis['52w Low'] = stock_data_analysis['52w Low'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else x)

    context.update(
      {
      'timestamp': '00:00:00',
      'stock_data_is_ready': True,
      'stock_data_analysis': stock_data_analysis.to_dict(orient='records')
      }
    )    
    return render_template('index.html', **context)

if __name__ == '__main__':
    app.run(port=9008, debug=True)
