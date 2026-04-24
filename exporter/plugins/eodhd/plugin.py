"""EODHD plugin — fetches fundamentals from eodhd.com and writes to InfluxDB."""

import time
from typing import Any, Dict, List

from plugins.base import PluginBase
from plugins.eodhd.client import EodhdClient
from plugins.eodhd.writer import EodhdInfluxWriter


class Plugin(PluginBase):
    """Pull EODHD fundamental data and push to InfluxDB."""

    @property
    def name(self) -> str:
        return "eodhd"

    def run(self, args: Dict[str, Any]) -> None:
        api_key = args.get("api_key")
        if not api_key:
            self.logger.error("No 'api_key' provided in plugin args")
            return

        # Plugin-level influxdb config takes precedence; loader injects
        # global influxdb config as a fallback.
        influx_cfg = args.get("influxdb") or {}

        # Support both flat keys (url/token/org/bucket) from global config
        # and prefixed keys (influxdb_url/...) from plugin-specific config
        url = influx_cfg.get("url") or influx_cfg.get("influxdb_url")
        token = influx_cfg.get("token") or influx_cfg.get("influxdb_token")
        org = influx_cfg.get("org") or influx_cfg.get("influxdb_org")
        bucket = influx_cfg.get("bucket") or influx_cfg.get("influxdb_bucket")

        if not all([url, token, org, bucket]):
            self.logger.error("Incomplete influxdb config in plugin args")
            return

        tickers: List[str] = args.get("tickers") or []
        exchange: str = args.get("exchange", "US")
        delay: float = args.get("delay_between_tickers", 1.0)

        client = EodhdClient(api_key=api_key)
        writer = EodhdInfluxWriter(
            url=str(url),
            token=str(token),
            org=str(org),
            bucket=str(bucket),
        )

        try:
            if tickers:
                self._process_tickers(client, writer, tickers, exchange, delay)
            else:
                self.logger.info("No tickers specified, fetching exchange symbol list")
                symbols = client.get_exchange_symbols(exchange)
                ticker_list = [s.get("Code") for s in symbols if s.get("Code")]
                self.logger.info(f"Found {len(ticker_list)} symbols on {exchange}")
                self._process_tickers(client, writer, ticker_list, exchange, delay)
        finally:
            writer.close()

        self.logger.info("EODHD plugin run complete")

    def _process_tickers(
        self,
        client: EodhdClient,
        writer: EodhdInfluxWriter,
        tickers: List[str],
        exchange: str,
        delay: float,
    ) -> None:
        total = len(tickers)
        success = 0
        errors = 0

        for i, ticker in enumerate(tickers, 1):
            self.logger.info(f"[{i}/{total}] Fetching fundamentals for {ticker}")
            try:
                data = client.get_fundamentals(ticker, exchange)

                if data.get("ETF_Data") or (data.get("General") or {}).get("Type") == "ETF":
                    self.logger.info(f"Skipping ETF: {ticker}")
                    continue

                writer.write_fundamentals_snapshot(data)

                financials = data.get("Financials") or {}
                if financials:
                    writer.write_financial_statements(ticker, financials, "quarterly")
                    writer.write_financial_statements(ticker, financials, "annual")

                earnings = data.get("Earnings") or {}
                if earnings:
                    writer.write_earnings(ticker, earnings)

                success += 1

            except Exception as e:
                self.logger.error(f"Error processing {ticker}: {e}")
                errors += 1

            if i < total:
                time.sleep(delay)

        self.logger.info(
            f"Finished: {success} succeeded, {errors} failed, "
            f"{total - success - errors} skipped"
        )
