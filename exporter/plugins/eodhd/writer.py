"""Write EODHD fundamental data to InfluxDB."""

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger("plugin.eodhd.writer")

TAG_FIELDS = {"ticker", "symbol", "exchange", "sector", "industry", "currency_code", "type"}

SKIP_FIELDS = {"_raw"}

STRING_FIELDS = {"name", "description", "address", "cusip", "ipo_date"}


def _safe_float(val: Any) -> float | None:
    """Convert to float, returning None for non-finite or non-numeric values."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class EodhdInfluxWriter:
    """Writes EODHD fundamental snapshots and financial statements to InfluxDB."""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.org = org
        self.bucket = bucket
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        logger.info(f"EODHD InfluxDB writer: {url}, bucket={bucket}")

    def write_fundamentals_snapshot(self, data: Dict[str, Any]) -> None:
        """Write a single ticker's fundamental highlights as a point.

        Measurement: ``eodhd_fundamentals``
        Tags: ticker, exchange, sector, industry, type
        Fields: market_cap, pe_ratio, eps, dividend_yield, etc.
        """
        general = data.get("General") or {}
        highlights = data.get("Highlights") or {}
        valuation = data.get("Valuation") or {}
        technicals = data.get("Technicals") or {}
        shares = data.get("SharesStats") or {}
        analyst = data.get("AnalystRatings") or {}

        ticker = general.get("Code", "UNKNOWN")
        now = datetime.now(timezone.utc)

        point = Point("eodhd_fundamentals").time(now, WritePrecision.S)

        point = point.tag("ticker", ticker)
        point = point.tag("exchange", general.get("Exchange", ""))
        point = point.tag("sector", general.get("Sector", ""))
        point = point.tag("industry", general.get("Industry", ""))
        point = point.tag("type", general.get("Type", ""))

        point = point.field("name", general.get("Name", ""))

        field_map = {
            "market_cap": highlights.get("MarketCapitalization"),
            "market_cap_mln": highlights.get("MarketCapitalizationMln"),
            "ebitda": highlights.get("EBITDA"),
            "pe_ratio": highlights.get("PERatio"),
            "peg_ratio": highlights.get("PEGRatio"),
            "eps": highlights.get("EarningsShare"),
            "eps_estimate_current_year": highlights.get("EPSEstimateCurrentYear"),
            "eps_estimate_next_year": highlights.get("EPSEstimateNextYear"),
            "book_value": highlights.get("BookValue"),
            "dividend_share": highlights.get("DividendShare"),
            "dividend_yield": highlights.get("DividendYield"),
            "profit_margin": highlights.get("ProfitMargin"),
            "operating_margin": highlights.get("OperatingMarginTTM"),
            "return_on_assets": highlights.get("ReturnOnAssetsTTM"),
            "return_on_equity": highlights.get("ReturnOnEquityTTM"),
            "revenue": highlights.get("RevenueTTM"),
            "revenue_per_share": highlights.get("RevenuePerShareTTM"),
            "gross_profit": highlights.get("GrossProfitTTM"),
            "diluted_eps": highlights.get("DilutedEpsTTM"),
            "quarterly_earnings_growth": highlights.get("QuarterlyEarningsGrowthYOY"),
            "quarterly_revenue_growth": highlights.get("QuarterlyRevenueGrowthYOY"),
            "trailing_pe": valuation.get("TrailingPE"),
            "forward_pe": valuation.get("ForwardPE"),
            "price_sales": valuation.get("PriceSalesTTM"),
            "price_book": valuation.get("PriceBookMRQ"),
            "enterprise_value": valuation.get("EnterpriseValue"),
            "enterprise_value_revenue": valuation.get("EnterpriseValueRevenue"),
            "enterprise_value_ebitda": valuation.get("EnterpriseValueEbitda"),
            "beta": technicals.get("Beta"),
            "week_52_high": technicals.get("52WeekHigh"),
            "week_52_low": technicals.get("52WeekLow"),
            "day_50_ma": technicals.get("50DayMA"),
            "day_200_ma": technicals.get("200DayMA"),
            "shares_outstanding": shares.get("SharesOutstanding"),
            "shares_float": shares.get("SharesFloat"),
            "short_ratio": shares.get("ShortRatio"),
            "short_percent_outstanding": shares.get("ShortPercentOutstanding"),
            "short_percent_float": shares.get("ShortPercentFloat"),
            "analyst_target_price": analyst.get("TargetPrice"),
            "analyst_strong_buy": analyst.get("StrongBuy"),
            "analyst_buy": analyst.get("Buy"),
            "analyst_hold": analyst.get("Hold"),
            "analyst_sell": analyst.get("Sell"),
            "analyst_strong_sell": analyst.get("StrongSell"),
        }

        for key, val in field_map.items():
            fval = _safe_float(val)
            if fval is not None:
                point = point.field(key, fval)

        self.write_api.write(bucket=self.bucket, org=self.org, record=point)
        logger.info(f"Wrote eodhd_fundamentals snapshot for {ticker}")

    def write_financial_statements(
        self, ticker: str, financials: Dict[str, Any], period: str
    ) -> None:
        """Write quarterly or annual financial statement rows.

        Measurement: ``eodhd_financials``
        Tags: ticker, period ('quarterly' | 'annual'), statement
        Fields: numeric financial line items
        """
        statements = {
            "balance_sheet": financials.get("Balance_Sheet") or {},
            "cash_flow": financials.get("Cash_Flow") or {},
            "income_statement": financials.get("Income_Statement") or {},
        }

        period_key = "quarterly" if period == "quarterly" else "yearly"
        points: List[Point] = []

        for stmt_name, stmt_data in statements.items():
            entries = stmt_data.get(period_key) or {}
            if not isinstance(entries, dict):
                continue

            for date_key, details in entries.items():
                if not isinstance(details, dict):
                    continue
                try:
                    filing_date = details.get("filing_date") or date_key
                    ts = datetime.strptime(filing_date[:10], "%Y-%m-%d").replace(
                        hour=16, tzinfo=timezone.utc
                    )
                except (ValueError, TypeError):
                    continue

                p = (
                    Point("eodhd_financials")
                    .tag("ticker", ticker)
                    .tag("period", period)
                    .tag("statement", stmt_name)
                    .time(ts, WritePrecision.S)
                )

                field_count = 0
                for field_name, field_val in details.items():
                    if field_name in ("date", "filing_date", "currency_symbol"):
                        continue
                    fval = _safe_float(field_val)
                    if fval is not None:
                        p = p.field(field_name, fval)
                        field_count += 1

                if field_count > 0:
                    points.append(p)

        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info(
                f"Wrote {len(points)} eodhd_financials points for {ticker} ({period})"
            )

    def write_earnings(self, ticker: str, earnings: Dict[str, Any]) -> None:
        """Write earnings history to InfluxDB.

        Measurement: ``eodhd_earnings``
        Tags: ticker
        Fields: epsActual, epsEstimate, epsDifference, surprisePercent
        """
        history = earnings.get("History") or {}
        if not isinstance(history, dict):
            history = {}

        points: List[Point] = []
        for date_key, entry in history.items():
            if not isinstance(entry, dict):
                continue
            try:
                ts = datetime.strptime(date_key[:10], "%Y-%m-%d").replace(
                    hour=16, tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                continue

            p = Point("eodhd_earnings").tag("ticker", ticker).time(ts, WritePrecision.S)
            field_count = 0
            for fname in ("epsActual", "epsEstimate", "epsDifference", "surprisePercent"):
                fval = _safe_float(entry.get(fname))
                if fval is not None:
                    p = p.field(fname, fval)
                    field_count += 1
            if field_count > 0:
                points.append(p)

        if points:
            self.write_api.write(bucket=self.bucket, org=self.org, record=points)
            logger.info(f"Wrote {len(points)} eodhd_earnings points for {ticker}")

    def close(self):
        self.client.close()
