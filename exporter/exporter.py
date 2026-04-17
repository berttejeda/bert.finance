#!/usr/bin/env python

import argparse
import logging
import sys
import time

from btconfig import Config

from lib.fetcher import batch_download_history, fetch_ticker_data
from lib.writer import InfluxWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("exporter")


def load_config(config_path="config.yaml"):
    """Load and return the app config via btconfig."""
    return Config(config_file_uri=config_path, templatized=True).read()


def run_once(config):
    """Fetch data for all tickers and write to InfluxDB."""
    tickers = config.get("tickers", [])
    settings = config.get("settings", {})
    influx_cfg = config.get("influxdb", {})

    delay = settings.get("delay_between_tickers", 2)
    period = settings.get("history_period", "1y")

    writer = InfluxWriter(
        url=influx_cfg["url"],
        token=influx_cfg["token"],
        org=influx_cfg["org"],
        bucket=influx_cfg["bucket"],
    )

    try:
        history = batch_download_history(tickers, period=period)
        results = []
        for ticker in tickers:
            df = history.get(ticker)
            if df is None or df.empty:
                logger.warning(f"No history for {ticker}, skipping")
                continue
            logger.info(f"Processing {ticker}")
            data = fetch_ticker_data(ticker, df, delay=delay)
            results.append(data)
        writer.write_batch(results, history_map=history)
    finally:
        writer.close()

    logger.info("Run complete")


def main():
    parser = argparse.ArgumentParser(description="Stock data exporter to InfluxDB")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously at the configured interval",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    interval = config.get("settings", {}).get("interval_minutes", 30)

    if args.loop:
        logger.info(f"Running in loop mode, interval={interval}m")
        while True:
            try:
                run_once(config)
            except Exception as e:
                logger.error(f"Run failed: {e}", exc_info=True)
            logger.info(f"Sleeping {interval} minutes")
            time.sleep(interval * 60)
    else:
        run_once(config)


if __name__ == "__main__":
    main()
