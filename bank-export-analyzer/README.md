# Bank Export Analyzer

A powerful Python tool to automatically categorize and validade your personal bank transaction exports.

## Overview

This project provides a script, `expense_analysis.py`, which reads bank export CSV files, categorizes transactions based on customizable regex patterns, and generates insightful reports and visualizations.

For a detailed explanation of how the script works under the hood, please see [Script Logic Documentation](SCRIPT_LOGIC.md).

## Features

-   **Automated Categorization**: Uses regex patterns defined in `categories.yaml`.
-   **Visualizations**: Generates monthly stacked bar charts to track spending trends.
-   **Flexible Input**: Supports various CSV formats (UTF-8, Latin1, BOM).
-   **CLI Interface**: Easy-to-use command line arguments.
-   **Smart Savings Tracking**: Automatically calculates net savings by deducting transfers *from* savings (configurable).
-   **Interactive Training**: Helps you categorize unknown transactions.
-   **Flexible Charting**: Control chart size, saving, and label display.

## Getting Started

1.  **Generate Sample Data** (Optional):
    If you don't have a bank export file yet, you can generate a sample one to test the script.
    ```bash
    python generate_sample_data.py
    ```
    This will create `sample_transactions.csv`.

2.  **Configure Categories**:
    The script uses `config.yaml` to define your transaction categories. A default configuration is provided, but you should customize it to match your spending.
    ```yaml
    # config.yaml
    categories:
      Food:
        - uber eats
        - starbucks
      Transport:
        - uber
        - shell
    ```

3.  **Run Analysis**:
    ```bash
    python expense_analysis.py --input-file sample_transactions.csv
    ```

4.  **View Results**:
    Check the `reports/` directory for:
    -   `cleaned_expenses.csv`
    -   `summary.csv`
    -   `expense_chart.png` (if `--save-chart-image` is used)

## Savings Logic

The script includes specific logic to calculate your **net** savings:
1.  **ToSavings**: Transactions categorized as `ToSavings` are treated as positive expenses (money leaving your checking account).
2.  **FromSavings / Deductions**:
    -   Credit transactions matching a specific pattern (default: `Funds.*Transfer.*From.*Share.*0000`) are treated as **negative expenses**.
    -   These are automatically assigned to the `ToSavings` category.
    -   **Result**: The `ToSavings` total represents the *net* amount transferred to savings (Transfers To - Transfers From).
    
    You can configure this pattern in `config.yaml`:
    ```yaml
    savings_deduction_pattern: "Transfer.*From.*Savings"
    ```
    Or via command line: `--savings-deduction-pattern "regex"`

## CLI Usage

-   **Standard Analysis**:
    ```bash
    python expense_analysis.py --input-file sample_transactions.csv
    ```

-   **Save Chart Image**:
    ```bash
    python expense_analysis.py --input-file sample_transactions.csv --save-chart-image
    ```
    Saves the generated chart to `reports/expense_chart.png`.

-   **Custom Configuration**:
    ```bash
    python expense_analysis.py --input-file sample_transactions.csv --config my_config.yaml
    ```
    Defaults to `config.yaml`.

-   **Interactive Plot**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --plot
    ```
    Displays the generated chart in a window.

-   **List Categorized Transactions**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --show-matched-categories-only
    ```
    Prints a sorted list of categorized transactions to the terminal and exits.

-   **Filter by Date Range**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --start 2023-01-01 --end 2023-01-31
    ```
    Analyzes only transactions within the specified date range.

-   **Train Categories**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --train-categories
    ```
    Interactively helps you categorize "Other" transactions by grouping similar descriptions.

-   **Filter by Category**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --filter "Food"
    ```
    Analyzes only transactions in the "Food" category.

-   **Filter by Year**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --year 2023
    ```
    Analyzes transactions for the entire year of 2023.

-   **Filter by Month**:
    ```bash
    python expense_analysis.py --input-file AccountHistory.csv --month 2023-05
    ```
    python expense_analysis.py --input-file sample_transactions.csv --month 2023-05
    ```
    Analyzes transactions for May 2023.

-   **Chart Options**:
    ```bash
    python expense_analysis.py -f file.csv --show-labels --chart-width 20 --chart-height 12
    ```
    - `--show-labels`: Adds value labels to chart segments (hides overlapping ones).
    - `--chart-width`, `--chart-height`: Sets figure size in inches.