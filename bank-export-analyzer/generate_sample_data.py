#!/usr/bin/env python

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(filename="sample_transactions.csv", num_rows=50):
    # Categories and likely descriptions
    vendors = {
        'Housing': ['RENT PAYMENT', 'MORTGAGE SERVICER'],
        'Food': ['SAFEWAY #1234', 'MCDONALDS 992', 'WHOLE FOODS MKT', 'UBER EATS'],
        'Transport': ['SHELL OIL 123455', 'CHEVRON 0022', 'UBER TRIP 8483'],
        'Utilities': ['PG&E WEB PAYMENT', 'CITY WATER DEPT'],
        'GeneralGoods': ['TARGET T-2608', 'AMZN Mktp US', 'WAL-MART #992'],
        'Entertainment': ['NETFLIX.COM', 'SPOTIFY AB'],
        'Healthcare': ['CVS PHARMACY #9929', 'KAISER PERM'],
    }
    
    data = []
    start_date = datetime(2026, 1, 1)
    
    for i in range(num_rows):
        # Pick a random category and vendor
        category = np.random.choice(list(vendors.keys()))
        vendor = np.random.choice(vendors[category])
        
        # Add some noise to description sometimes
        if np.random.random() > 0.5:
            description = f"{vendor}  POS ID: {np.random.randint(10000, 99999)}"
        else:
            description = vendor
            
        date = start_date + timedelta(days=np.random.randint(0, 60))
        
        # Random debit amount
        debit = round(np.random.uniform(5.0, 500.0), 2)
        credit = np.nan
        
        # Occasionally make it an income/credit
        if np.random.random() > 0.9:
            debit = np.nan
            credit = round(np.random.uniform(1000.0, 3000.0), 2)
            description = "PAYROLL DEPOSIT"
            
        data.append({
            'Account Number': '123456789',
            'Post Date': date.strftime('%m/%d/%Y'),
            'Check': np.nan,
            'Description': description,
            'Debit': debit,
            'Credit': credit,
            'Status': 'Posted',
            'Balance': 0 # Placeholder
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Generated {filename} with {num_rows} rows.")

if __name__ == "__main__":
    generate_sample_data()
