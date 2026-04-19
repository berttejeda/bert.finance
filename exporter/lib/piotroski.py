import logging

logger = logging.getLogger("piotroski")


def _safe_get(df, label, col_idx=0, default=None):
    """Safely extract a value from a yfinance financial DataFrame.

    Args:
        df: DataFrame where rows are line items and columns are dates.
        label: Row label (e.g. 'Net Income').
        col_idx: Column index (0 = most recent period).
        default: Fallback if the label or column doesn't exist.

    Returns:
        The scalar value or default.
    """
    try:
        if label in df.index and col_idx < len(df.columns):
            val = df.loc[label].iloc[col_idx]
            if val is not None and str(val) != "nan":
                return float(val)
    except Exception:
        pass
    return default


def calc_piotroski_score(ticker_obj):
    """Calculate the Piotroski score (0-9) using yfinance financial statements.

    This mirrors the logic from the polygon/stocks-analysis Patrioski class,
    with corrections:
      - ROA uses total assets (not current assets).
      - Robust fallbacks when data is missing.

    Args:
        ticker_obj: A yfinance Ticker object.

    Returns:
        Dict with 'piotroski_score' (int or None) and 'piotroski_details' (dict).
    """
    symbol = ticker_obj.ticker
    try:
        inc = ticker_obj.income_stmt
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
    except Exception as e:
        logger.warning(f"Could not fetch financials for {symbol}: {e}")
        return {"piotroski_score": None, "piotroski_details": {}}

    if inc.empty or bs.empty or cf.empty:
        logger.warning(f"Insufficient financial data for {symbol}")
        return {"piotroski_score": None, "piotroski_details": {}}

    if len(inc.columns) < 2 or len(bs.columns) < 2:
        logger.warning(f"Need at least 2 annual periods for {symbol}")
        return {"piotroski_score": None, "piotroski_details": {}}

    score = 0
    details = {}

    # --- Current year (col 0) and previous year (col 1) ---

    # CR1: Net Income > 0
    ni = _safe_get(inc, "Net Income", 0)
    if ni is not None:
        cr1 = 1 if ni > 0 else 0
    else:
        cr1 = 0
    details["CR1_net_income"] = ni
    score += cr1

    # CR2: ROA > 0  (net_income / average total assets)
    ta_cy = _safe_get(bs, "Total Assets", 0)
    ta_py = _safe_get(bs, "Total Assets", 1)
    if ni is not None and ta_cy and ta_py:
        avg_assets = (ta_cy + ta_py) / 2
        roa = ni / avg_assets if avg_assets else 0
        cr2 = 1 if roa > 0 else 0
    else:
        roa = None
        cr2 = 0
    details["CR2_roa"] = round(roa, 6) if roa is not None else None
    score += cr2

    # CR3: Operating Cash Flow > 0
    ocf = _safe_get(cf, "Operating Cash Flow", 0)
    if ocf is not None:
        cr3 = 1 if ocf > 0 else 0
    else:
        cr3 = 0
    details["CR3_ocf"] = ocf
    score += cr3

    # CR4: Cash flow from operations > Net Income (quality of earnings)
    if ocf is not None and ni is not None:
        cr4 = 1 if ocf > ni else 0
    else:
        cr4 = 0
    details["CR4_quality"] = cr4
    score += cr4

    # CR5: Decrease in long-term debt
    ltd_cy = _safe_get(bs, "Long Term Debt", 0, 0)
    ltd_py = _safe_get(bs, "Long Term Debt", 1, 0)
    ltd_delta = ltd_cy - ltd_py
    cr5 = 1 if ltd_delta < 0 else 0
    details["CR5_ltd_delta"] = ltd_delta
    score += cr5

    # CR6: Current ratio improved
    ca_cy = _safe_get(bs, "Current Assets", 0)
    cl_cy = _safe_get(bs, "Current Liabilities", 0)
    ca_py = _safe_get(bs, "Current Assets", 1)
    cl_py = _safe_get(bs, "Current Liabilities", 1)
    if ca_cy and cl_cy and ca_py and cl_py and cl_cy != 0 and cl_py != 0:
        cr_cy = ca_cy / cl_cy
        cr_py = ca_py / cl_py
        cr6 = 1 if (cr_cy - cr_py) > 0 else 0
        details["CR6_current_ratio_delta"] = round(cr_cy - cr_py, 4)
    else:
        cr6 = 0
        details["CR6_current_ratio_delta"] = None
    score += cr6

    # CR7: No new shares issued
    shares_cy = _safe_get(inc, "Basic Average Shares", 0) or \
                _safe_get(inc, "Diluted Average Shares", 0)
    shares_py = _safe_get(inc, "Basic Average Shares", 1) or \
                _safe_get(inc, "Diluted Average Shares", 1)
    if shares_cy is not None and shares_py is not None:
        shares_delta = shares_cy - shares_py
        cr7 = 1 if shares_delta <= 0 else 0
    else:
        shares_delta = None
        cr7 = 0
    details["CR7_shares_delta"] = shares_delta
    score += cr7

    # CR8: Gross margin improved
    gp_cy = _safe_get(inc, "Gross Profit", 0)
    rev_cy = _safe_get(inc, "Total Revenue", 0)
    gp_py = _safe_get(inc, "Gross Profit", 1)
    rev_py = _safe_get(inc, "Total Revenue", 1)
    if gp_cy and rev_cy and gp_py and rev_py and rev_cy != 0 and rev_py != 0:
        gm_cy = gp_cy / rev_cy
        gm_py = gp_py / rev_py
        cr8 = 1 if (gm_cy - gm_py) > 0 else 0
        details["CR8_gross_margin_delta"] = round(gm_cy - gm_py, 6)
    else:
        cr8 = 0
        details["CR8_gross_margin_delta"] = None
    score += cr8

    # CR9: Asset turnover improved
    ta_py2 = _safe_get(bs, "Total Assets", 2) if len(bs.columns) > 2 else None
    if rev_cy and ta_cy and ta_py and rev_py and ta_py2:
        at_cy = rev_cy / ((ta_cy + ta_py) / 2) if (ta_cy + ta_py) else 0
        at_py = rev_py / ((ta_py + ta_py2) / 2) if (ta_py + ta_py2) else 0
        cr9 = 1 if (at_cy - at_py) > 0 else 0
        details["CR9_asset_turnover_delta"] = round(at_cy - at_py, 6)
    else:
        cr9 = 0
        details["CR9_asset_turnover_delta"] = None
    score += cr9

    logger.info(f"Piotroski score for {symbol}: {score}/9")
    return {"piotroski_score": score, "piotroski_details": details}
