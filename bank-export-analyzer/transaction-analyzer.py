#!/usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import yaml
import re
from categorization_trainer import CategorizationTrainer

class ExpenseCategorizer:
    def __init__(self, config_file="config.yaml", mode="expenses"):
        self.mode = mode
        self.categories = self._load_config(config_file)

    def _load_config(self, config_file):
        """Loads category regex patterns from a YAML file."""
        if not os.path.exists(config_file):
            print(f"Warning: Config file '{config_file}' not found. Using default empty config.")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
                
            # Mode-specific loading
            section = config.get(self.mode, {})
            # 'categories' might be a dict (expenses) or list of dicts (income)
            raw_categories = section.get('categories', {})
            
            # Normalize to dict: {category: [patterns]}
            normalized_categories = {}
            
            if isinstance(raw_categories, list):
                # Handle list of dicts: [{Category: [patterns]}, ...]
                for item in raw_categories:
                    if isinstance(item, dict):
                        for cat, patterns in item.items():
                            normalized_categories[cat] = patterns
            elif isinstance(raw_categories, dict):
                # Handle standard dict
                normalized_categories = raw_categories
            else:
                # Fallback check for old structure if not found in section
                # If mode is expenses, maybe checking root? 
                # For now, strict adherence to new structure is safer, but fallback is nice.
                if self.mode == 'expenses':
                     return config.get('categories', config)

            return normalized_categories

        except Exception as e:
            print(f"Error loading config file: {e}")
            return {}

    def categorize(self, description):
        """Categorizes a transaction based on regex patterns."""
        description = str(description).lower()
        
        # Check against each category in the config
        for category, patterns in self.categories.items():
            # Ensure patterns is a list, even if it's a single string
            if isinstance(patterns, str):
                patterns = [patterns]
            
            if not patterns: continue

            for pattern in patterns:
                try:
                    if re.search(pattern, description, re.IGNORECASE):
                        return category
                except re.error as e:
                    print(f"Regex error for pattern '{pattern}': {e}")
        return "Other"

def analyze_transactions(input_file, config_file="config.yaml", plot=False, show_matched_categories_only=False, start_date=None, end_date=None, train_categories=False, category_filter=None, show_labels=False, chart_width=15, chart_height=10, savings_deduction_pattern=None, save_chart_image=False, mode="expenses"):
    """
    Reads the transaction CSV, categorizes, and generates a summary report and chart.
    """
    # Configure pandas display options to show full content
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 1000)
    
    print(f"Starting {mode.upper()} analysis...")

    # Load config early to get settings
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")

    # Determine savings_deduction_pattern
    # Priority: CLI arg > Config file > Default
    # Determine savings_deduction_pattern
    # Only applicable in 'expenses' mode
    if mode == 'expenses':
        # Priority: CLI arg > Config file > Default
        if savings_deduction_pattern is None:
            # Look inside expenses section
            expenses_config = config.get('expenses', {})
            # Also check root for backward compatibility or if structure matches old style
            savings_deduction_pattern = expenses_config.get('savings_deduction_pattern', config.get('savings_deduction_pattern', r"Transfer.*From.*Savings"))
        print(f"Using savings deduction pattern: '{savings_deduction_pattern}'")
    else:
        print("Income mode: Savings deduction logic disabled.")

    print(f"Using savings deduction pattern: '{savings_deduction_pattern}'")
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    print(f"Reading {input_file}...")
    try:
        if input_file.lower().endswith(('.xls', '.xlsx')):
            try:
                df = pd.read_excel(input_file)
            except ImportError as e:
                print(f"Error reading Excel file: {e}")
                print("Please install 'openpyxl' (for .xlsx) or 'xlrd' (for .xls).")
                return
        else:
            # Try utf-8-sig first to handle BOM
            try:
                df = pd.read_csv(input_file, encoding='utf-8-sig')
            except UnicodeDecodeError:
                print("utf-8-sig failed, trying latin1...")
                df = pd.read_csv(input_file, encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Normalize columns
    df.columns = [c.strip() for c in df.columns]

    # Check for Date column
    date_col = None
    for col in df.columns:
        if 'date' in col.lower():
            date_col = col
            break
    
    if date_col:
        print(f"Found Date column: {date_col}")
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['Month'] = df[date_col].dt.to_period('M')

        # Filter by date range if provided
        if start_date:
            print(f"Filtering transactions from {start_date}...")
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            print(f"Filtering transactions up to {end_date}...")
            df = df[df[date_col] <= pd.to_datetime(end_date)]
            
        print(f"Transactions after filtering: {len(df)}")
        if df.empty:
            print("No transactions found in the specified date range.")
            return
    else:
        print("Warning: No Date column found. Monthly analysis will be skipped.")
        if start_date or end_date:
             print("Warning: Date filtering requested but no Date column found. Ignoring filter.")

    # Check for required columns
    # If Description is missing, maybe it's named differently?
    if 'Description' not in df.columns:
        # Try case-insensitive matching
        col_map = {c.lower(): c for c in df.columns}
        if 'description' in col_map:
            df.rename(columns={col_map['description']: 'Description'}, inplace=True)
            print("Found 'Description' column with different casing.")
        else:
            print(f"Error: CSV is missing 'Description' column.")
            print(f"Found columns: {df.columns.tolist()}")
            return
    
    # Handle Debit/Credit logic
    # Normalize 'Debit' and 'Credit' columns if present
    if 'Debit' in df.columns and 'Credit' in df.columns:
        df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
        df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)
        
        if mode == 'expenses':
            # Taking Debits as expenses
            df['Amount'] = df['Debit']
            
            # Specific Logic: Deduct Credits matching the savings_deduction_pattern from expenses
            # We handle this by making them negative expenses
            mask = df['Description'].str.contains(savings_deduction_pattern, case=False, regex=True) & (df['Credit'] > 0)
            
            if mask.any():
                print(f"Found {mask.sum()} credits matching the savings_deduction_pattern to deduct.")
                df.loc[mask, 'Amount'] = -df.loc[mask, 'Credit']
        
        elif mode == 'income':
            # Taking Credits as income
            df['Amount'] = df['Credit']
            # No savings deduction logic for income

        # Filter out 0 value transactions (unless they are the negative ones we just created for expenses)
        df = df[df['Amount'] != 0].copy()
        
    elif 'Amount' in df.columns:
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        if mode == 'expenses':
             # Assume positive amounts are expenses if only one column? Or filter for negative?
             # Standardizing: usually bank csvs with one column have - for expenses.
             # If - for expenses, we might want to flip them to positive for the chart?
             # Existing logic didn't flip, so charts likely showed negatives or inputs were absolute.
             # The previous code seemed to just take Amount.
             pass
    else:
        # Try case-insensitive Amount
        col_map = {c.lower(): c for c in df.columns}
        if 'amount' in col_map:
             df['Amount'] = pd.to_numeric(df[col_map['amount']], errors='coerce').fillna(0)
        else:
            print("Error: Could not find 'Debit'/'Credit' or 'Amount' columns.")
            return

    # Initialize Categorizer
    print(f"Loading categories from {config_file} for mode '{mode}'...")
    categorizer = ExpenseCategorizer(config_file, mode=mode)

    # Categorize
    print(f"Categorizing {len(df)} transactions...")
    df['Category'] = df['Description'].apply(categorizer.categorize)

    # Force categorization for the negative savings transactions if they were marked as 'Other'
    # These should fall into 'ToSavings' to reduce the total
    # Pattern is already determined at start of function

    if mode == 'expenses':
        mask = df['Description'].str.contains(savings_deduction_pattern, case=False, regex=True) & (df['Amount'] < 0)
        if mask.any():
            print("Assigning negative 'From Savings' transactions to 'ToSavings' category...")
            df.loc[mask, 'Category'] = 'ToSavings'

    # Filter by Category if requested
    if category_filter:
        print(f"Filtering transactions for category: {category_filter}")
        df = df[df['Category'].str.lower() == category_filter.lower()]
        print(f"Transactions after category filtering: {len(df)}")
        if df.empty:
            print(f"No transactions found for category '{category_filter}'.")
            return

    if show_matched_categories_only:
        matched_df = df[['Category', 'Description']].sort_values(by='Category')
        print("\n--- Matched Categories ---")
        print(matched_df.to_string(index=False))
        return

    # Debug: Check categorization stats
    category_counts = df['Category'].value_counts()
    print("\nCategorization Stats:")
    print(category_counts)
    
    if "Other" in category_counts and category_counts["Other"] == len(df):
        print("\nWARNING: All transactions categorized as 'Other'. Checking sample descriptions...")
        print(df[['Description', 'Category']].head(5))

    # Interactive Training
    if train_categories:
        print("\n--- Starting Categorization Training ---")
        other_df = df[df['Category'] == 'Other'].copy()
        
        if other_df.empty:
            print("No 'Other' transactions found to train on.")
        else:
            trainer = CategorizationTrainer(other_df, config_file, mode=mode)
            trainer.interactive_labeling()
            
            # Re-load categorizer and re-categorize to show impact
            print("\nRe-categorizing with new patterns...")
            categorizer = ExpenseCategorizer(config_file, mode=mode)
            df['Category'] = df['Description'].apply(categorizer.categorize)
            print("\nUpdated Categorization Stats:")
            print(df['Category'].value_counts())

    # Summarize Total
    summary = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    
    print(f"\nTotal {mode.capitalize()} Summary:")
    print(summary)

    # Create output directory
    output_folder = "reports"
    os.makedirs(output_folder, exist_ok=True)
    
    # Save reports
    cleaned_file = os.path.join(output_folder, "cleaned_expenses.csv")
    summary_file = os.path.join(output_folder, "summary.csv")
    chart_file = os.path.join(output_folder, "expense_chart.png")

    df.to_csv(cleaned_file, index=False)
    summary.to_csv(summary_file)

    # Visualization
    print("\nGenerating visualization...")
    
    if 'Month' in df.columns:
        # Monthly Stacked Bar Chart
        monthly_summary = df.groupby(['Month', 'Category'])['Amount'].sum().unstack().fillna(0)
        
        # Sort months chronologically
        monthly_summary = monthly_summary.sort_index()
        
        print("\nMonthly Summary:")
        print(monthly_summary)

        # Sort categories by total amount (descending) so the legend is sorted
        # This also puts larger categories at the bottom of the stack
        category_totals_sort = monthly_summary.sum(axis=0)
        sorted_categories = category_totals_sort.sort_values(ascending=False).index
        monthly_summary = monthly_summary[sorted_categories]
        
        plt.figure(figsize=(chart_width, chart_height))
        ax = plt.gca()
        # Add a horizontal line at 0 for reference
        ax.axhline(0, color='black', linewidth=0.8)
        monthly_summary.plot(kind='bar', stacked=True, ax=ax, colormap='tab20', edgecolor='black', linewidth=0.5)
        plt.title(f"Monthly {mode.capitalize()} by Category")
        plt.xlabel("Month")
        plt.ylabel("Amount ($)")
        plt.xticks(rotation=45)
        plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add labels to each bar segment
        all_annotations = []
        if show_labels:
            for c in ax.containers:
                # value is the height of the segment. Allow negative values.
                labels = [f'{v.get_height():.0f}' if v.get_height() != 0 else '' for v in c]
                annotations = ax.bar_label(c, labels=labels, label_type='center', fontsize=8)
                all_annotations.extend(zip(c, annotations))

        # Add total labels at the top of each bar
        # Total is calculated as sum of absolute values
        abs_totals = monthly_summary.abs().sum(axis=1)
        # Position is top of positive stack + offset
        positive_totals = monthly_summary[monthly_summary > 0].sum(axis=1)

        # Add grand total to the legend itself
        # Use algebraic sum for the Period Total to account for negative savings/income correctly
        grand_total = monthly_summary.sum().sum()
        
        # Calculate totals per category to add to legend labels
        # Use algebraic sum here too to show Net for each category
        category_totals = monthly_summary.sum(axis=0)
        
        # Calculate monthly averages
        # We average over the number of months in the period
        # If a category had 0 expenses in a month, it still counts as a month (0 value)
        # monthly_summary already has 0s filled for missing category-months
        category_averages = monthly_summary.mean(axis=0)

        # Create dummy handles for the "Totals" section
        # Importing patches might be needed, or we can use plot() with visible=False
        # We need to get current handles/labels first
        handles, labels = ax.get_legend_handles_labels()
        
        # Update labels with amounts and averages
        new_labels = []
        for label in labels:
            if label in category_totals:
                total_amount = category_totals[label]
                avg_amount = category_averages[label]
                # Escape dollar signs for Matplotlib
                new_labels.append(f"{label} (\\${total_amount:,.2f} | Avg: \\${avg_amount:,.2f})")
            else:
                new_labels.append(label)
        labels = new_labels
        
        # Add a Spacer and a Header
        import matplotlib.lines as mlines
        empty_handle = mlines.Line2D([], [], color='white', marker='None', linestyle='None', label='')
        
        handles.append(empty_handle)
        labels.append("")
        
        handles.append(empty_handle)
        labels.append(f"Total {mode.capitalize()}")

        handles.append(empty_handle)
        labels.append(f"Period Total: \\${grand_total:,.2f}")

        # Add Average Monthly Outflow section
        # Calculate average total monthly spend (net)
        avg_monthly_outflow = monthly_summary.sum(axis=1).mean()
        
        handles.append(empty_handle)
        labels.append("")

        handles.append(empty_handle)
        labels.append(f"Average Monthly {mode.capitalize()}")

        handles.append(empty_handle)
        labels.append(f"Monthly Average: \\${avg_monthly_outflow:,.2f}")

        legend = plt.legend(handles=handles, labels=labels, title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Make the title bold
        plt.setp(legend.get_title(), fontweight='bold')
        
        # Make the section headers bold
        for text in legend.get_texts():
            if text.get_text() in [f"Total {mode.capitalize()}", f"Average Monthly {mode.capitalize()}"]:
                text.set_fontweight('bold')

        plt.tight_layout()

        # Overlap Prevention Logic
        # Draw the canvas to calculate text sizes
        fig = plt.gcf()
        fig.canvas.draw()
        
        if show_labels:
            renderer = fig.canvas.get_renderer()
            for bar, text in all_annotations:
                if not text.get_text():
                    continue
                
                # Get the bounding box of the text and bar in display coordinates
                text_bbox = text.get_window_extent(renderer)
                bar_bbox = bar.get_window_extent(renderer)
                
                # Check height (with a small buffer)
                # If text is taller than bar, hide it
                if text_bbox.height > bar_bbox.height:
                    text.set_visible(False)
        
        # Use bbox_inches='tight' to ensure external text/legend is not cut off
        if save_chart_image:
            plt.savefig(chart_file, bbox_inches='tight')
            print(f"Monthly stacked chart saved to: {chart_file}")
        else:
            print("Chart image saving skipped (use --save-chart-image to save).")
        
        if plot:
            print("Displaying plot...")
            plt.show()
    else:
        # Fallback to simple bar chart
        plt.figure(figsize=(10, 6))
        summary.plot(kind='bar', ax=plt.gca(), color='skyblue')
        plt.title(f"Total {mode.capitalize()} Summary by Category")
        plt.xlabel("Category")
        plt.ylabel("Amount ($)")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.tight_layout()

        if save_chart_image:
            plt.savefig(chart_file)
            print(f"Total summary chart saved to: {chart_file}")
        else:
            print("Chart image saving skipped (use --save-chart-image to save).")

        if plot:
            print("Displaying plot...")
            plt.show()

    print(f"\nAnalysis complete!")
    print(f"Cleaned data saved to: {cleaned_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze bank export files.")
    parser.add_argument("--input-file","-f", required=True, help="Path to the input CSV file")
    parser.add_argument("--config","-c", default="config.yaml", help="Path to the configuration YAML file")
    parser.add_argument("--plot", action="store_true", help="Display the plot interactively")
    parser.add_argument("--show-matched-categories-only", "-mo", action="store_true", help="Display categorized transactions and exit")
    parser.add_argument("--start", "-s", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", help="End date (YYYY-MM-DD)")
    parser.add_argument("--train-categories", "-t", action="store_true", help="Interactively train categories using clustering")
    parser.add_argument("--filter", help="Filter transactions by specific category")
    parser.add_argument("--year", "-y", help="Filter by year (e.g. 2023). Mutually exclusive with --start/--end")
    parser.add_argument("--month", "-m", help="Filter by month (e.g. 2023-05). Mutually exclusive with --start/--end/--year")

    parser.add_argument("--show-labels", action="store_true", help="Show data labels on the chart (requires --plot or chart generation)")
    parser.add_argument("--chart-width", type=float, default=15, help="Width of the chart in inches (default: 15)")
    parser.add_argument("--chart-height", type=float, default=10, help="Height of the chart in inches (default: 10)")
    parser.add_argument("--savings-deduction-pattern", help="Regex pattern for savings deductions (overrides config)")
    parser.add_argument("--save-chart-image", action="store_true", help="Save the chart to an image file")
    
    # Mutually exclusive group for mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--debits", dest='mode', action='store_const', const='expenses', help="Analyze expenses (Debits). Default.")
    group.add_argument("--credits", dest='mode', action='store_const', const='income', help="Analyze income (Credits).")
    parser.set_defaults(mode='expenses')
    
    args = parser.parse_args()

    # handle shortcuts
    if args.year:
        if args.start or args.end or args.month:
            parser.error("Argument --year cannot be used with --start, --end, or --month")
        args.start = f"{args.year}-01-01"
        args.end = f"{args.year}-12-31"
        
    if args.month:
        if args.start or args.end or args.year:
            parser.error("Argument --month cannot be used with --start, --end, or --year")
        try:
            period = pd.Period(args.month, freq='M')
            args.start = str(period.start_time.date())
            args.end = str(period.end_time.date())
        except ValueError:
             parser.error("Invalid format for --month. Use YYYY-MM.")
    try:
        analyze_transactions(args.input_file, args.config, args.plot, args.show_matched_categories_only, args.start, args.end, args.train_categories, args.filter, args.show_labels, args.chart_width, args.chart_height, args.savings_deduction_pattern, args.save_chart_image, args.mode)
    except KeyboardInterrupt:
        print("\nProcess interrupted.")
