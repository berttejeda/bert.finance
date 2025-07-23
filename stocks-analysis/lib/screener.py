import logging
import requests
import statistics
from datetime import datetime, timedelta
from polygon import RESTClient

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('screener')

class Screener:
    def __init__(self, **kwargs):
        self.base_url = kwargs['base_url']
        self.polygon_api_key = kwargs['polygon_api_key']
        self.client = RESTClient(self.polygon_api_key)

    def get_historical_prices(self, ticker, days=14):
        """Get historical daily close prices for RSI & trend."""
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days*2)  # buffer for non-trading days
    
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}?adjusted=true&sort=desc&limit=100&apiKey={self.polygon_api_key}"
        res = requests.get(url).json()
        closes = [bar["c"] for bar in res.get("results", [])][-days:]
        return closes
    
    def calculate_rsi(self, closes, period=14):
        """Relative Strength Index (RSI)"""
        if len(closes) < period:
            return None
        gains = [closes[i] - closes[i-1] for i in range(1, len(closes)) if closes[i] > closes[i-1]]
        losses = [-1 * (closes[i] - closes[i-1]) for i in range(1, len(closes)) if closes[i] < closes[i-1]]
        avg_gain = sum(gains)/period if gains else 0.01
        avg_loss = sum(losses)/period if losses else 0.01
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def get_option_snapshot(self, ticker):
        """Get option snapshot with Greeks & IV."""
        options_chain = []
        for o in self.client.list_snapshot_options_chain(
                ticker,
                params={
                    "order": "asc",
                    "sort": "ticker",
                },
        ):
            options_chain.append(o)
        calls = [o for o in options_chain if o.details.contract_type == "call"]
        otm_calls = [c for c in calls if c.greeks.delta is not None and 0.3 <= c.greeks.delta <= 0.45 and c.implied_volatility > 0.3 and c.open_interest > 1000]
        return otm_calls
    
    def is_uptrend(self, prices):
        """Basic trend check: higher recent closes & price above 50 MA."""
        if len(prices) < 14:
            return False
        ma_50 = statistics.mean(prices)
        return prices[-1] > ma_50 and prices[-1] > prices[0]
    
    def screen(self, ticker):
        logger.info(f"Running stock screener for {ticker}...")
        data_obj = {
                    "ticker": ticker,
                    "result": False,
        }
        try:
            prices = self.get_historical_prices(ticker)
            rsi = self.calculate_rsi(prices)
            is_uptrend = self.is_uptrend(prices)
            if (rsi or rsi <= 70) and is_uptrend:
                otm_calls = self.get_option_snapshot(ticker)
                data_obj['price'] =  prices[-1]
                data_obj['prices'] = prices
                data_obj['rsi'] = rsi
                data_obj['result'] = True
                data_obj['otm_calls'] = otm_calls
                if otm_calls:
                    best_call = sorted(otm_calls, key=lambda x: x.implied_volatility, reverse=True)[0]
                    data_obj['rsi'] = f"{rsi:.2f}"
                    data_obj['iv'] = f"{best_call.implied_volatility * 100:.2f}"
                    data_obj['delta'] = f"{best_call.greeks.delta:.2f}"
                    data_obj['open_interest'] = best_call.open_interest
                    data_obj['expiration'] = best_call.details.expiration_date
                    data_obj['strike'] = best_call.details.strike_price
        except Exception as e:
            logger.error(f"Error screening {ticker}: {e}")
        logger.info(f'Finished screening {ticker}')
        return data_obj