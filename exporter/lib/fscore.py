import logging

logger = logging.getLogger("fscore")


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


def _current_ratio(bs, col_idx):
    ca = _safe_get(bs, "Current Assets", col_idx)
    cl = _safe_get(bs, "Current Liabilities", col_idx)
    if ca is not None and cl is not None and cl != 0:
        return ca / cl
    return None


def _gross_margin(inc, col_idx):
    gp = _safe_get(inc, "Gross Profit", col_idx)
    rev = _safe_get(inc, "Total Revenue", col_idx)
    if gp is not None and rev is not None and rev != 0:
        return gp / rev
    return None


def _asset_turnover(inc, bs, col_idx):
    """Asset turnover = Revenue / Beginning-of-year total assets.

    Beginning-of-year total assets for period col_idx is approximated
    by the total assets reported in the prior period (col_idx + 1).
    """
    rev = _safe_get(inc, "Total Revenue", col_idx)
    beg_ta = _safe_get(bs, "Total Assets", col_idx + 1)
    if rev is not None and beg_ta is not None and beg_ta != 0:
        return rev / beg_ta
    return None


def calc_fscore(ticker_obj):
    """Calculate the Piotroski F-Score (0-9) using ratio-based methodology.

    This follows the academically precise definitions from Piotroski (2000):
      - ROA = Net Income / Total Assets
      - CFO = Operating Cash Flow / Beginning-of-year Total Assets
      - Accrual = CFO - ROA (both as ratios, not raw values)
      - ΔLeverage = (LTD_cy - LTD_py) / avg(TA_cy, TA_py)

    Reference implementation: F-Score-Calculator by SarmadTanveer.

    Args:
        ticker_obj: A yfinance Ticker object.

    Returns:
        Dict with 'fscore' (int or None) and 'fscore_details' (dict of ratios).
    """
    symbol = ticker_obj.ticker
    try:
        inc = ticker_obj.income_stmt
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
    except Exception as e:
        logger.warning(f"Could not fetch financials for {symbol}: {e}")
        return {"fscore": None, "fscore_details": {}}

    if inc.empty or bs.empty or cf.empty:
        logger.warning(f"Insufficient financial data for {symbol}")
        return {"fscore": None, "fscore_details": {}}

    if len(inc.columns) < 2 or len(bs.columns) < 2:
        logger.warning(f"Need at least 2 annual periods for {symbol}")
        return {"fscore": None, "fscore_details": {}}

    details = {}
    score = 0

    # --- Current year = col 0, Previous year = col 1 ---

    # ROA = Net Income / Total Assets
    ni_cy = _safe_get(inc, "Net Income", 0)
    ta_cy = _safe_get(bs, "Total Assets", 0)
    ni_py = _safe_get(inc, "Net Income", 1)
    ta_py = _safe_get(bs, "Total Assets", 1)

    roa_cy = (ni_cy / ta_cy) if (ni_cy is not None and ta_cy and ta_cy != 0) else None
    roa_py = (ni_py / ta_py) if (ni_py is not None and ta_py and ta_py != 0) else None

    # CFO = Operating Cash Flow / Beginning-of-year Total Assets
    ocf_cy = _safe_get(cf, "Operating Cash Flow", 0)
    beg_ta_cy = _safe_get(bs, "Total Assets", 1)  # beginning of current year ≈ end of prior year
    cfo_ratio = (ocf_cy / beg_ta_cy) if (ocf_cy is not None and beg_ta_cy and beg_ta_cy != 0) else None

    # F1: ROA > 0
    if roa_cy is not None:
        f1 = 1 if roa_cy > 0 else 0
    else:
        f1 = 0
    details["F1_roa"] = round(roa_cy, 6) if roa_cy is not None else None
    score += f1

    # F2: CFO > 0
    if cfo_ratio is not None:
        f2 = 1 if cfo_ratio > 0 else 0
    else:
        f2 = 0
    details["F2_cfo"] = round(cfo_ratio, 6) if cfo_ratio is not None else None
    score += f2

    # F3: ΔROA > 0
    if roa_cy is not None and roa_py is not None:
        droa = roa_cy - roa_py
        f3 = 1 if droa > 0 else 0
    else:
        droa = None
        f3 = 0
    details["F3_droa"] = round(droa, 6) if droa is not None else None
    score += f3

    # F4: Accrual — CFO > ROA (both as ratios)
    if cfo_ratio is not None and roa_cy is not None:
        f4 = 1 if cfo_ratio > roa_cy else 0
    else:
        f4 = 0
    details["F4_accrual"] = f4
    score += f4

    # F5: ΔLeverage < 0 — (LTD_cy - LTD_py) / avg(TA_cy, TA_py)
    ltd_cy = _safe_get(bs, "Long Term Debt", 0, 0)
    ltd_py = _safe_get(bs, "Long Term Debt", 1, 0)
    if ta_cy and ta_py and (ta_cy + ta_py) != 0:
        dleverage = (ltd_cy - ltd_py) / ((ta_cy + ta_py) / 2)
        f5 = 1 if dleverage < 0 else 0
    else:
        dleverage = None
        f5 = 0
    details["F5_dleverage"] = round(dleverage, 6) if dleverage is not None else None
    score += f5

    # F6: ΔLiquidity > 0 — current_ratio_cy - current_ratio_py
    cr_cy = _current_ratio(bs, 0)
    cr_py = _current_ratio(bs, 1)
    if cr_cy is not None and cr_py is not None:
        dliquid = cr_cy - cr_py
        f6 = 1 if dliquid > 0 else 0
    else:
        dliquid = None
        f6 = 0
    details["F6_dliquid"] = round(dliquid, 4) if dliquid is not None else None
    score += f6

    # F7: No equity offered — shares_cy <= shares_py
    shares_cy = _safe_get(inc, "Basic Average Shares", 0) or \
                _safe_get(inc, "Diluted Average Shares", 0)
    shares_py = _safe_get(inc, "Basic Average Shares", 1) or \
                _safe_get(inc, "Diluted Average Shares", 1)
    if shares_cy is not None and shares_py is not None:
        eq_offered = shares_cy - shares_py
        f7 = 1 if eq_offered <= 0 else 0
    else:
        eq_offered = None
        f7 = 0
    details["F7_eq_offered"] = eq_offered
    score += f7

    # F8: ΔMargin > 0 — gross_margin_cy - gross_margin_py
    gm_cy = _gross_margin(inc, 0)
    gm_py = _gross_margin(inc, 1)
    if gm_cy is not None and gm_py is not None:
        dmargin = gm_cy - gm_py
        f8 = 1 if dmargin > 0 else 0
    else:
        dmargin = None
        f8 = 0
    details["F8_dmargin"] = round(dmargin, 6) if dmargin is not None else None
    score += f8

    # F9: ΔTurnover > 0 — asset_turnover_cy - asset_turnover_py
    at_cy = _asset_turnover(inc, bs, 0)
    at_py = _asset_turnover(inc, bs, 1)
    if at_cy is not None and at_py is not None:
        dturn = at_cy - at_py
        f9 = 1 if dturn > 0 else 0
    else:
        dturn = None
        f9 = 0
    details["F9_dturn"] = round(dturn, 6) if dturn is not None else None
    score += f9

    logger.info(f"F-Score for {symbol}: {score}/9")
    return {"fscore": score, "fscore_details": details}
