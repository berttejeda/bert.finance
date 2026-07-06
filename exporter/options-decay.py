import numpy as np
from scipy.stats import norm
from datetime import date, timedelta
import argparse

def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return max(0, S - K) if option_type == 'call' else max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def main():
    parser = argparse.ArgumentParser(description="HIMS Options Strategy Simulator")
    parser.add_argument("--mode", choices=['simple', '3-leg'], default='simple', 
                        help="Strategy: 'simple' (Short 29C) or '3-leg' (Collar + Kicker)")
    parser.add_argument("--days", type=int, default=30, help="Days from today to simulate")
    parser.add_argument("--price", type=float, default=24.50, help="Target stock price for single point analysis")
    args = parser.parse_args()

    # --- CONSTANTS ---
    N_CONTRACTS = 7
    SHARES = N_CONTRACTS * 100
    R, SIGMA = 0.045, 0.75
    START_DATE, EXPIRY_DATE = date(2026, 5, 14), date(2026, 11, 20)
    TOTAL_DAYS = (EXPIRY_DATE - START_DATE).days
    T_REM = max(0, TOTAL_DAYS - args.days) / 365.0
    
    # --- PRICES ---
    PRICES = [20.0, 22.0, 24.5, 27.0, 30.0]

    print(f"\n{'='*50}\nMODE: {args.mode.upper()} | SIMULATION DATE: {START_DATE + timedelta(days=args.days)}\n{'='*50}")
    print(f"{'Stock Price':<12} | {'Option Value':<15} | {'Net Profit':<12}")
    print("-" * 50)

    for S in PRICES:
        if args.mode == 'simple':
            # Original Trade: Short 29 Call
            init_credit = 3096.22
            val = black_scholes(S, 29.0, T_REM, R, SIGMA, 'call')
            profit = init_credit - (val * SHARES + 4.78)
        else:
            # 3-Leg: Sell 30C, Buy 25C, Buy 20P
            # Initial setup cost (at current price $24.50)
            c30_0 = black_scholes(24.50, 30.0, TOTAL_DAYS/365.0, R, SIGMA, 'call')
            c25_0 = black_scholes(24.50, 25.0, TOTAL_DAYS/365.0, R, SIGMA, 'call')
            p20_0 = black_scholes(24.50, 20.0, TOTAL_DAYS/365.0, R, SIGMA, 'put')
            net_setup = (c30_0 - c25_0 - p20_0) * SHARES
            
            # Future value
            c30 = black_scholes(S, 30.0, T_REM, R, SIGMA, 'call')
            c25 = black_scholes(S, 25.0, T_REM, R, SIGMA, 'call')
            p20 = black_scholes(S, 20.0, T_REM, R, SIGMA, 'put')
            val = (c30 - c25 - p20) * SHARES
            profit = val - net_setup

        print(f"${S:<11.2f} | ${abs(val):<14.2f} | ${profit:<11.2f}")

if __name__ == "__main__":
    main()