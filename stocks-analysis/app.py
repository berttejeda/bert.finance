from datetime import datetime
from flask import Flask, render_template, jsonify
from tasks import get_cache_key, cache, tickers, earnings_by_day, get_current_vix
from io import StringIO

import argparse
import logging
import pandas as pd
import re

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

@app.route('/')
def home():
    context = {
        'stock_data_is_ready': False,
        'current_vix': get_current_vix(),
        'earnings_by_day': earnings_by_day
    }
    key = get_cache_key(tickers)
    cached = cache.get(key)
    if cached:
        stock_data_analysis = pd.read_json(StringIO(cached.decode('utf-8')))
        stock_data_analysis = stock_data_analysis.dropna(subset=['Price'])
        # stock_data_analysis['Price'] = stock_data_analysis['Price'].round(2)
        stock_data_analysis = stock_data_analysis.dropna(subset=['CompletedAt'])
        stock_data_analysis = stock_data_analysis.dropna(subset=['P/E'])
        task_duration = stock_data_analysis.dropna(subset=['Duration'])
        task_duration_in_minutes = task_duration['Duration'].sum() / 60
        last_cache_refresh = pd.to_datetime(stock_data_analysis['CompletedAt']).max().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f'Task duration was {round(task_duration_in_minutes,2)}')
        context.update(
          {
          'last_cache_refresh': last_cache_refresh,
          'current_vix': get_current_vix(),
          'task_duration_in_minutes': task_duration_in_minutes,
          'stock_data_is_ready': True,
          'stock_data_analysis': stock_data_analysis.to_dict(orient='records'),
          'total_tickers': len(stock_data_analysis)
          }
        )         
        return render_template('index.html', **context)
    else:
        return render_template('index.html', **context)

@app.route('/data-status')
def data_status():
    key = get_cache_key(tickers)
    is_ready = {'ready': bool(cache.exists(key))}
    return jsonify(is_ready)

if __name__ == '__main__':
    app.run(host=host_address, port=host_port, debug=debug)
