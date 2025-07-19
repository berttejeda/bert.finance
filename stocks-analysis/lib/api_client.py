from collections import Counter
from datetime import date, datetime, timedelta
from lib.chart_utils import fig_to_base64, plot_empty_data
from lib.patrioski import Patrioski
from polygon import RESTClient

import base64
import io
import logging
import math
import matplotlib
import matplotlib.pyplot as plt
import requests

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('api-client')

# Configure matplotlib
matplotlib.use('Agg')

import pandas as pd

class PolygonTicker:
    def __init__(self, **kwargs):
        symbol = kwargs['symbol']
        self.polygon_api_key = kwargs['polygon_api_key']
        self.quiver_api_key = kwargs['quiver_api_key']
        self.earnings_endpoint = "https://api.polygon.io/benzinga/v1/earnings"
        self.sic_data = kwargs['sic_data']
        self.symbol = symbol.upper()
        self.client = RESTClient(self.polygon_api_key)
        self.default_period = 7
        self.today = datetime.now()
        self.ticker_snapshot = self.client.get_snapshot_ticker(
            "stocks",
            self.symbol
        )
        self.sma_50 = self.get_sma(50)
        self.sma_100 = self.get_sma(100)
        self.sma_150 = self.get_sma(150)
        self.sma_200 = self.get_sma(200)
        self.aggregates = self.get_aggregates()
        self.aggregates_past_year = self.get_aggregates_past_year()
        self.last_trade = self.client.get_last_trade(self.symbol)
        self.price_data_last_trading_month = self.get_price_last_trading_month()
        self.rsi_data = self.calculate_rsi()
        self.current_price = f"{self.last_trade.price:.2f}"
        self.vroc_data = self.get_vroc(self.aggregates)
        self.macd_data = self.calculate_macd_signals(self.aggregates)
        self.news_data = self.get_news(10,14)
        self.sma_stock_signal = self.get_sma_stock_signal(self.sma_50, self.sma_150)
        previous_year = date.today().year - 1
        current_year = date.today().year
        self.current_year_str = f"{current_year:04d}"
        self.previous_year_str = f"{previous_year:04d}"
        self.ticker_details_current_year = self.client.get_ticker_details(self.symbol)
        self.ticker_details_previous_year = self.client.get_ticker_details(self.symbol, date=f"{self.previous_year_str}-12-31")
        self.fundamentals = self.get_fundamentals()
        patrioski = Patrioski(
            client=self.client,
            ticker=self.symbol,
            current_year_str=self.current_year_str,
            fundamentals=self.fundamentals,
            previous_year_str=self.previous_year_str,
            ticker_details_current_year=self.ticker_details_current_year,
            ticker_details_previous_year=self.ticker_details_previous_year
        )
        self.patrioski_data = patrioski.calculate_score()
        self.company_info = self.fetch_company_info()
        self.earnings_data = self.get_current_and_last_year_earnings()
        self.next_earnings_date = self.get_next_earnings_date()
        fifty_two_week_highs_lows = self.get_fifty_two_week_highs_lows()
        self.fifty_two_week_low = f"{fifty_two_week_highs_lows['low']:.2f}"
        self.fifty_two_week_high = f"{fifty_two_week_highs_lows['high']:.2f}"
        trailing_pe = self.fundamentals[0].financials.income_statement.basic_earnings_per_share.value
        if not trailing_pe or math.isnan(trailing_pe):
            logger.warning(f'Trailing PE for {self.symbol} is Not a Number!')
            self.trailing_pe = ''
        else:
            self.trailing_pe = f"{trailing_pe:.2f}"
        self.bollinger_stock_data = self.get_bollinger_stock_data()
        self.senator_trade_data = self.get_senator_trades()

    def get_fifty_two_week_highs_lows(self):
        # Define the date range (last 52 weeks from today)
        end_date = self.today
        start_date = end_date - timedelta(weeks=52)

        # Get aggregate bars for the specified date range
        aggs = []
        for a in self.client.list_aggs(
                ticker=self.symbol,
                multiplier=1,
                timespan="day",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000,  # Adjust limit if necessary
        ):
            aggs.append(a)

        # Initialize high and low values
        fifty_two_week_high = float('-inf')
        fifty_two_week_low = float('inf')

        # Loop through data to find high and low
        for agg in aggs:
            if agg.high > fifty_two_week_high:
                fifty_two_week_high = agg.high
            if agg.low < fifty_two_week_low:
                fifty_two_week_low = agg.low
        data_obj = {
            "low": fifty_two_week_low,
            "high": fifty_two_week_high
        }
        return data_obj

    def get_fundamentals(self):
        financials = []
        for f in self.client.vx.list_stock_financials(
                order='desc',
                ticker=self.symbol,
                filing_date_gte=f"{self.previous_year_str}-01-01",
                sort="filing_date",
        ):
            financials.append(f)
        return financials

    def get_fundamentals_annual(self):
        financials = []
        for f in self.client.vx.list_stock_financials(
                order='desc',
                ticker=self.symbol,
                timeframe="annual",
                sort="filing_date",
                limit=1
        ):
            financials.append(f)
        return financials

    def get_earnings_for_year(self, year: int):
        """Fetch earnings (confirmed + projected) for a given calendar year."""
        params = {
            "ticker": self.symbol,
            "fiscal_year": year,
            "sort": "date.asc",
            "limit": 50000,
            "apiKey": self.polygon_api_key
        }

        resp = requests.get(self.earnings_endpoint, params=params)
        resp.raise_for_status()
        result = resp.json().get("results", [])
        return result

    def get_current_and_last_year_earnings(self):
        """Return a dict with earnings for the current and previous year."""
        today = datetime.utcnow().date()
        current_year = today.year
        prev_year = current_year - 1

        current = self.get_earnings_for_year(current_year)
        previous = self.get_earnings_for_year(prev_year)

        earnings_data = {
            "current_year": current,
            "previous_year": previous
        }

        return earnings_data

    def get_next_earnings_date(self, confirmed_only: bool = False):
        """Return the next earnings date from today."""
        today = datetime.utcnow().date().isoformat()

        params = {
            "ticker": self.symbol,
            "date.gte": today,
            "sort": "date.asc",
            "limit": 1,
            "apiKey": self.polygon_api_key
        }

        if confirmed_only:
            params["date_status"] = "confirmed"

        resp = requests.get(self.earnings_endpoint, params=params)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        return results[0] if results else None

    # Example usage:
    if __name__ == "__main__":
        data = get_current_and_last_year_earnings(ticker="AAPL")
        print(f"Apple earnings this year: {len(data['current_year'])} records")
        print(f"Apple earnings last year: {len(data['previous_year'])} records")

    def fetch_earnings_data(self):
        URL = "https://api.polygon.io/benzinga/v1/earnings"
        PARAMS = {
            "order": "asc",
            "ticker": self.symbol,
            "date": "",
            "limit": 100,
            "sort": "date",
            "apiKey": self.polygon_api_key
        }

        response = requests.get(URL, params=PARAMS)

        if response.status_code == 200:
            data = response.json()
            print(data)
        else:
            print(f"Request failed with status {response.status_code}")
            print(response.text)

    def fetch_company_info(self):
        industry = self.sic_data.codes[self.ticker_details_current_year.sic_code].industry_title
        website_url = self.ticker_details_current_year.homepage_url
        website_display = website_url if website_url != '#' else ''
        company_info_markdown = f"""
### Company Info
- **Summary**: {self.ticker_details_current_year.description or 'No summary available.'}
- **Industry**: {industry}
- **Website**: [{website_display}]({website_url})
    """
        data_obj = {
            "info": self.ticker_details_current_year,
            "summary": company_info_markdown,
            "industry": industry,
        }
        return data_obj

    def get_aggregates(self, from_date=None, to_date=None):
        aggs = []
        if not from_date:
            from_date = (self.today + timedelta(days=-37)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = self.today.strftime('%Y-%m-%d')
        for a in self.client.list_aggs(
                self.symbol,
                1,
                "day",
                from_date,
                to_date,
                adjusted="true",
                sort="asc",
                limit=100,
        ):
            aggs.append(a)
        return aggs


    def get_aggregates_past_year(self, from_date=None, to_date=None):
        aggs = []
        if not from_date:
            from_date = (self.today + timedelta(weeks=-52)).strftime('%Y-%m-%d')
        if not to_date:
            to_date = self.today.strftime('%Y-%m-%d')
        for a in self.client.list_aggs(
                self.symbol,
                1,
                "day",
                from_date,
                to_date,
                adjusted="true",
                sort="asc",
                limit=100,
        ):
            aggs.append(a)
        return aggs

    def get_sma(self, window, limit=10):
        sma_data = self.client.get_sma(
        ticker=self.symbol,
        timespan="day",
        adjusted="true",
        window=window,
        series_type="close",
        order="desc",
        limit="10",
        )
        sma_values = [v.value for v in sma_data.values]
        sma = sum(sma_values) / len(sma_values)
        return int(sma)

    def get_daily_bars(self, days=7):
        end = datetime.utcnow()
        start = end - timedelta(days=days)

        bars = []
        for bar in self.client.list_aggs(
            ticker=self.symbol,
            multiplier=1,
            timespan="day",
            from_=start.strftime('%Y-%m-%d'),
            to=end.strftime('%Y-%m-%d'),
            limit=500
        ):
            bars.append({
                "timestamp": datetime.utcfromtimestamp(bar.timestamp / 1000),
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })

        df = pd.DataFrame(bars).set_index("timestamp")
        return df

    def get_price_last_trading_month(self):
        data = pd.DataFrame(self.aggregates)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms').dt.date
        chart = fig_to_base64(self.plot_price_last_trading_month(data))
        chart_description = f"""
        {self.symbol} - Price over the last trading month
        """
        data_obj = {
            'Chart': chart,
            'Chart Description': chart_description,
        }
        return data_obj

    def plot_price_last_trading_month(self, data):
        # Plot the data
        fig = plt.figure(figsize=(12, 2))
        plt.plot(data['timestamp'], data['close'], marker='o', linestyle='-')
        plt.title(f"{self.symbol} - Price over the last trading month")
        plt.xlabel("Date")
        plt.ylabel("Close Price (USD)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_stock_news_data(self, data):
        # Group by formatted timestamp and count occurrences
        counts = data.groupby('Date').size()
        fig = plt.figure(figsize=(12, 2))  # Adjust figure size as needed
        plt.plot(counts.index, counts.values, marker='o', linestyle='-')  # Line plot
        # plt.fill_between(counts.index, counts.values, alpha=0.5)
        plt.xlabel('Time')
        plt.ylabel('Count')
        plt.title(f'{self.symbol} News Articles Over Time')
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        return fig

    def create_news_markdown(self, data):
        markdown_content = f'''<h1 style="text-align:center;">{self.symbol} News</h1>
        
{data.to_markdown()}
'''
        return markdown_content

    def get_news(self, limit=10, period=None):
        date_period = period or self.default_period
        time_delta = self.today - timedelta(days=date_period)
        cutoff_date = time_delta.strftime("%Y-%m-%d")
        news_data = self.client.list_ticker_news(
            self.symbol,
        params={"published_utc.gte": cutoff_date},
        order="desc",
        limit=limit
        )
        # Convert generator to list of dicts
        news_list = [{
            "sentiment": Counter([s.sentiment for s in article.insights]).most_common(1)[0][0],
            "published_utc": article.published_utc,
            "title": article.title,
            "author": article.author,
            "article_url": article.article_url,
            "description": article.description,
        } for article in news_data]
        # Create DataFrame
        if news_list:
            news_data_obj = pd.DataFrame(news_list)
            news_data_obj['Date'] = pd.to_datetime(news_data_obj['published_utc'], utc=True)
            # overall_sentiment = Counter(sentiments).most_common(1)[0][0]
            overall_sentiment = Counter([sentiment for sentiment in news_data_obj['sentiment']]).most_common(1)[0][0]
            news_data_obj = news_data_obj.drop(columns=['published_utc'])
            chart = fig_to_base64(self.plot_stock_news_data(news_data_obj))
            chart_description = self.create_news_markdown(news_data_obj)
            data_obj = {
                'Chart': chart,
                'Chart Description': chart_description,
                'overall_sentiment': overall_sentiment,
            }
        else:
            logger.warning(f'News list is empty!')
            data_obj = {
                'Chart': fig_to_base64(plot_empty_data(self.symbol, f'No news data found for {self.symbol}')),
                'Chart Description': '',
                'overall_sentiment': ''
            }
        return data_obj

    def get_bollinger_stock_data(self, window=20, std_multiplier=2):
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
        df = pd.DataFrame(self.aggregates_past_year)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Convert timestamp to datetime
        df = df.set_index('timestamp')

        df['SMA'] = df['close'].rolling(window=window).mean()
        df['SD'] = df['close'].rolling(window=window).std()
        df['Upper'] = df['SMA'] + (std_multiplier * df['SD'])
        df['Lower'] = df['SMA'] - (std_multiplier * df['SD'])

        df['Signal'] = 0  # Initialize with no signal
        # Buy signal: Price crosses below the lower band
        df.loc[df['close'] < df['Lower'], 'Signal'] = 1
        # Sell signal: Price crosses above the upper band
        df.loc[df['close'] > df['Upper'], 'Signal'] = -1

        latest = df.iloc[-1]
        close = latest['close'].item()
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

    def calculate_rsi(self, window="14"):
        rsi_data = self.client.get_rsi(
            ticker=self.symbol,
            timespan="day",
            adjusted="true",
            window=str(window),
            series_type="close",
            order="desc",
            limit="10",
        )
        # Extract timestamps and RSI values
        rsi_timestamps = [item.timestamp for item in rsi_data.values]
        rsi_values = [item.value for item in rsi_data.values]
        rsi_chart = fig_to_base64(self.plot_rsi(rsi_timestamps, rsi_values))
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
            'Chart Description': rsi_chart_description
        }
        return data_obj

    def plot_rsi(self, timestamps, values):
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.plot(timestamps, values, label='RSI', color='purple')
        ax.axhline(70, linestyle='--', color='red')
        ax.axhline(30, linestyle='--', color='green')
        ax.set_title(f'{self.symbol} RSI Chart')
        ax.legend()
        return fig

    def get_vroc(self, aggregates, vroc_period=14):
        # Convert aggregate data to DataFrame
        data = pd.DataFrame([{
            "timestamp": pd.to_datetime(a.timestamp, unit="ms"),
            "volume": a.volume
        } for a in aggregates])

        # Sort by date (important for correct VROC calculation)
        data.sort_values("timestamp", inplace=True)

        # Calculate VROC
        data["VROC"] = ((data["volume"] - data["volume"].shift(vroc_period)) / data["volume"].shift(vroc_period)) * 100

        # Drop NaNs introduced by the shift
        data.dropna(inplace=True)

        vroc_signal_chart_description = """
VROC - Volume Rate of Change

This indicator measures the percentage change in trading volume over a specific period. 
A rising VROC could signal increasing interest in a stock.
        """
        data_obj = {
            'Chart': fig_to_base64(self.plot_vroc(data)),
            'Chart Description': vroc_signal_chart_description
        }
        return data_obj

    def plot_vroc(self, data):
        fig = plt.figure(figsize=(12, 2))
        plt.title(f'{self.symbol} Volume Rate of Change (VROC) Chart')
        plt.plot(data['VROC'], label='VROC', color='orange')
        return fig

    def calculate_macd_signals(self, aggregates):

        # Build DataFrame
        df = pd.DataFrame([{
            "timestamp": pd.to_datetime(a.timestamp, unit="ms"),
            "close": a.close
        } for a in aggregates])
        # Sort chronologically
        df.sort_values("timestamp", inplace=True)
        # Calculate MACD components
        df["EMA12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["Histogram"] = df["MACD"] - df["Signal"]
        macd_chart = fig_to_base64(self.plot_macd(df))
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
            'Chart Description': macd_chart_description
        }
        return data_obj

    def plot_macd(self, df):
        # signal = df.ewm(span=9, adjust=False).mean()
        fig, ax = plt.subplots(figsize=(12, 2))
        ax.plot(df.index, df["MACD"], label='MACD', color='blue')
        ax.plot(df.index, df["Signal"], label='Signal', color='orange')
        ax.set_title(f'{self.symbol} MACD Chart')
        ax.legend()
        return fig
        # Plotting with Matplotlib
        fig, ax = plt.subplots(figsize=(12, 2))
        # MACD and Signal Line
        ax.plot(df["timestamp"], df["MACD"], label="MACD", color="blue")
        ax.plot(df["timestamp"], df["Signal"], label="Signal Line", color="orange")
        # Histogram bars
        ax.bar(df["timestamp"], df["Histogram"], label="Histogram", color="gray", width=1)
        ax.set_title(f"{self.symbol} MACD Indicator")
        ax.xlabel("Date")
        ax.ylabel("MACD")
        ax.legend()
        ax.grid(True)
        ax.tight_layout()
        return fig

    def get_sma_stock_signal(self, ma_50, ma_150):

        ma_signal = ""
        if not isinstance(self.current_price, str):
            if self.current_price > self.sma_50 and self.current_price > self.sma_150:
                ma_signal = "Buy"
            elif self.current_price < self.sma_50 and self.current_price < self.sma_150:
                ma_signal = "Sell"
            else:
                ma_signal = "Hold"
        data_obj = {
            "Signal": ma_signal,
        }
        return data_obj

    def get_senator_trades(self):
        chart_description = f"""
    Senate Trades for {self.symbol}
    """
        try:
            if not self.quiver_api_key:
                raise('QUIVER_API_KEY not defined!')
            url = f"https://api.quiverquant.com/beta/historical/congresstrading/{self.symbol}"
            headers = {"Authorization": f"Bearer {self.quiver_api_key}"}
            response = requests.get(url, headers=headers, timeout=5)

            if response.status_code == 200:
                trades = response.json()

                rows = []
                for trade in trades:
                    name = trade.get('Representative')
                    action = trade.get('Transaction')
                    date = trade.get('TransactionDate')
                    if name and action and date:
                        rows.append({
                            'name': name,
                            'action': action,
                            'date': date.replace("-", "/")
                        })

                data = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["name", "action", "date"])
                time_delta = self.today - timedelta(weeks=4)
                cutoff_date = pd.to_datetime(time_delta)
                data['date'] = pd.to_datetime(data['date'], format="%Y/%m/%d")
                if not data['date'].empty:
                    data = data[data['date'] > cutoff_date].reset_index(drop=True)
                    chart = self.plot_senator_trades(data)
                    data_obj = {
                        'Chart': chart,
                        'Chart Description': chart_description
                    }
                    return data_obj
        except Exception as e:
            logger.error(f'Failed to fetch senate trade data for {self.symbol}, error was {e}')
        data_obj = {
            'Chart': fig_to_base64(plot_empty_data(self.symbol, f'No senate trade data available for {self.symbol} in the last 4 weeks')),
            'Chart Description': chart_description
        }
        return data_obj

    def plot_senator_trades(self, data):
        # Parse dates and extract relevant info
        stacked_counts = data.groupby(['name', 'action']).size().unstack(fill_value=0)
        # Generate plot
        fig, ax = plt.subplots(figsize=(12, 6))
        stacked_counts.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title('Trade Actions by Senator - last month')
        ax.set_xlabel('Senator')
        ax.set_ylabel('Number of Trades')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Convert plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.read()).decode('utf8')
        plt.close()
        return plot_url