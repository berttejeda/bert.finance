import logging
import pandas as pd

# Configure the logging system
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Create a logger
logger = logging.getLogger('patrioski-calc')

class Patrioski:
    def __init__(self, **kwargs):
        self.client = kwargs['client']
        self.ticker = kwargs['ticker']
        self.fundamentals = kwargs['fundamentals']
        self.current_year_str = kwargs['current_year_str']
        self.previous_year_str = kwargs['previous_year_str']
        self.ticker_details_current_year = kwargs['ticker_details_current_year']
        self.ticker_details_previous_year = kwargs['ticker_details_previous_year']
        self.description = "Piotroski score (0–9) indicating a firm's financial strength based on 9 accounting signals."
        self.default_data_obj = {
                "Description": self.description,
                "Score": '',
                "Data": pd.DataFrame()
            }

    def net_income(self, fin):
        return fin.financials.income_statement.net_income_loss.value

    def roa(self, financials, net_income):
        total_assets_previous_year = \
        [f.financials.balance_sheet.current_assets.value for f in financials if self.previous_year_str in f.filing_date][0]
        total_assets_current_year = \
        [f.financials.balance_sheet.current_assets.value for f in financials if self.current_year_str in f.filing_date][0]
        average_assets = (total_assets_previous_year + total_assets_current_year) / 2
        result = round(net_income / average_assets, 4) if average_assets else 0
        return result

    def ocf(self, fin):
        result = fin.financials.cash_flow_statement.net_cash_flow_from_operating_activities.value
        return result

    def ltdebt_delta(self, f0, f1):
        f0_long_term_debt = f0.financials.balance_sheet.long_term_debt.value
        f1_long_term_debt = f1.financials.balance_sheet.long_term_debt.value
        if all([f0_long_term_debt, f1_long_term_debt]):
            result = f0_long_term_debt - f1_long_term_debt
        else:
            result = 0
        return result

    def current_ratio_delta(self, f0, f1):
        curr0 = f0.financials.balance_sheet.current_assets.value / f0.financials.balance_sheet.current_liabilities.value
        curr1 = f1.financials.balance_sheet.current_assets.value / f1.financials.balance_sheet.current_liabilities.value
        result = round(curr0 - curr1, 2)
        return result

    def new_shares(self, curr, prev):
        result = curr.weighted_shares_outstanding - prev.weighted_shares_outstanding
        return result

    def gross_margin_delta(self, f0, f1):
        gp0 = f0.financials.income_statement.gross_profit.value
        rev0 = f0.financials.income_statement.revenues.value
        gp1 = f1.financials.income_statement.gross_profit.value
        rev1 = f1.financials.income_statement.revenues.value
        if not rev0 or not rev1:
            return 0
        result = (gp0 / rev0) - (gp1 / rev1)
        return result

    def asset_turnover_delta(self, f0, f1, f2):
        a0, a1, a2 = [f.financials.balance_sheet.current_assets.value for f in (f0, f1, f2)]
        rev0 = f0.financials.income_statement.revenues.value
        rev1 = f1.financials.income_statement.revenues.value
        at0 = rev0 / ((a0 + a1) / 2) if a0 and a1 else 0
        at1 = rev1 / ((a1 + a2) / 2) if a1 and a2 else 0
        result = round(at0 - at1, 2)
        return result

    def calculate_score(self):
        fundamentals = self.fundamentals
        if len(fundamentals) < 3:
            logger.warning(f"Not enough financial data to calculate Piotroski score for {self.ticker}")
            return self.default_data_obj
        f0 = [f for f in fundamentals if self.current_year_str in f.filing_date][0]
        f1 = [f for f in fundamentals if self.previous_year_str in f.filing_date][0]
        f2 = [f for f in fundamentals if self.previous_year_str in f.filing_date][1]

        raw = {}
        score = 0
        try:
            # CR1: Net Income > 0
            ni = self.net_income(f0)
            raw['CR1'] = ni
            score += 1 if ni > 0 else 0

            # CR2: ROA > 0
            r = self.roa(fundamentals, ni)
            raw['CR2'] = r
            score += 1 if r > 0 else 0

            # CR3: CFO > 0
            cash = self.ocf(f0)
            raw['CR3'] = cash
            score += 1 if cash > 0 else 0

            # CR4: CFO > Net Income
            raw['CR4'] = int(cash > ni)
            score += raw['CR4']

            # CR5: Decrease in long-term debt
            ltd = self.ltdebt_delta(f0, f1)
            raw['CR5'] = ltd
            score += 1 if ltd < 0 else 0

            # CR6: Current Ratio Improvement
            cr = self.current_ratio_delta(f0, f1)
            raw['CR6'] = cr
            score += 1 if cr > 0 else 0

            # CR7: No new shares issued
            shares = self.new_shares(self.ticker_details_current_year, self.ticker_details_previous_year)
            raw['CR7'] = shares
            score += 1 if shares <= 0 else 0

            # CR8: Improved gross margin
            gm = self.gross_margin_delta(f0, f1)
            raw['CR8'] = gm
            score += 1 if gm > 0 else 0

            # CR9: Improved asset turnover
            at = self.asset_turnover_delta(f0, f1, f2)
            raw['CR9'] = at
            score += 1 if at > 0 else 0

            df = pd.DataFrame([{
                'Symbol': self.ticker,
                'Name': self.ticker_details_current_year.name,
                **raw,
                'Score': score
            }])
            data_obj = {
                "Description": self.description,
                "Score": score,
                "Data": df
            }
            return data_obj
        except Exception as e:
            logger.warning(f'Failed to calculate patrioski score for {self.ticker}, error was {e}')
            return self.default_data_obj

