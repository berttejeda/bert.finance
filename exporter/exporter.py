#!/usr/bin/env python

import argparse
import logging
import sys
import time

from dotenv import load_dotenv
load_dotenv()

from btconfig import Config

from lib.fetcher import batch_download_history, batch_download_intraday, fetch_ticker_data
from lib.writer import InfluxWriter
from plugins.loader import load_plugins

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
    raw_tickers = config.get("tickers", [])
    tickers = list(dict.fromkeys(raw_tickers))  # deduplicate, preserve order
    if len(tickers) < len(raw_tickers):
        logger.warning(f"Removed {len(raw_tickers) - len(tickers)} duplicate ticker(s)")
    settings = config.get("settings", {})
    influx_cfg = config.get("influxdb", {})

    delay = settings.get("delay_between_tickers", 2)
    period = settings.get("history_period", "1y")
    intraday_period = settings.get("intraday_period", "5d")
    intraday_interval = settings.get("intraday_interval", "1m")

    writer = InfluxWriter(
        url=influx_cfg["url"],
        token=influx_cfg["token"],
        org=influx_cfg["org"],
        bucket=influx_cfg["bucket"],
    )

    try:
        history = batch_download_history(tickers, period=period)
        intraday = batch_download_intraday(tickers, period=intraday_period, interval=intraday_interval)
        results = []
        for ticker in tickers:
            df = history.get(ticker)
            if df is None or df.empty:
                logger.warning(f"No history for {ticker}, skipping")
                continue
            logger.info(f"Processing {ticker}")
            data = fetch_ticker_data(ticker, df, delay=delay)
            results.append(data)
        writer.write_batch(results, history_map=history, intraday_map=intraday)
    finally:
        writer.close()

    # --- Plugins ---
    run_plugins(config)

    logger.info("Run complete")


def run_plugins(config, only=None):
    """Load and execute enabled plugins.

    Args:
        config: Full parsed config dict.
        only: If set, run only the plugin with this name.
    """
    plugins = load_plugins(config)
    if not plugins:
        logger.debug("No enabled plugins found")
        return

    for plugin, plugin_args in plugins:
        if only and plugin.name != only:
            continue
        logger.info(f"=== Running plugin: {plugin.name} ===")
        try:
            plugin.run(plugin_args)
        except Exception as e:
            logger.error(f"Plugin '{plugin.name}' failed: {e}", exc_info=True)

    if only and not any(p.name == only for p, _ in plugins):
        logger.error(f"Plugin '{only}' not found or not enabled")


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
    parser.add_argument(
        "--plugin", "-p",
        type=str,
        default=None,
        help="Run only the named plugin (skip core export)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    interval = config.get("settings", {}).get("interval_minutes", 30)

    if args.plugin:
        # Plugin-only mode: skip core export, run just the named plugin
        logger.info(f"Running plugin-only mode: {args.plugin}")
        run_plugins(config, only=args.plugin)
    elif args.loop:
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
