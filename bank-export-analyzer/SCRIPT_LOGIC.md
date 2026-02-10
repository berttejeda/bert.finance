# Expense Analysis Script Logic

This document details the inner workings of `expense_analysis.py`, a tool designed to categorize and visualize bank transaction data.

## 1. Data Loading & Cleaning

The script uses `pandas` to read CSV files. It attempts to handle different file encodings automatically:
-   **UTF-8-SIG**: Handles BOM (Byte Order Mark) often found in Excel exports.
-   **Latin1**: Fallback encoding if UTF-8 fails.

### Column Normalization
-   All column names are stripped of whitespace.
-   The script looks for a `Description` column. If missing, it attempts case-insensitive matching (e.g., `description`).
-   It identifies `Debit` and `Credit` columns to calculate a unified `Amount` (Debits are positive expenses).
-   It detects date columns (containing "date" in the name) and parses them into datetime objects for time-series analysis.

## 2. Categorization Engine (`ExpenseCategorizer`)

Categorization is driven by a YAML configuration file (`categories.yaml` by default).

-   **Logic**: The `ExpenseCategorizer` class loads regex patterns from the YAML file.
-   **Process**:
    1.  The transaction description is converted to lowercase.
    2.  The script iterates through each category in the YAML.
    3.  It checks if any of the regex patterns for that category match the description (`re.search`).
    4.  The first match determines the category.
    5.  If no match is found, it defaults to "Other".

**Example YAML Structure:**
```yaml
Food:
  - uber eats
  - starbucks
Transport:
  - uber
  - shell
```

## 3. Visualization

The script uses `matplotlib` to generate charts.

-   **Monthly Stacked Bar Chart**: If a valid date column is found, expenses are grouped by `Month` and `Category`. This shows spending trends over time. Segments are labeled with their values if `--show-labels` is used.
-   **Simple Bar Chart**: If no date is found, a simple total expense by category bar chart is generated.
-   **Interactive Training**: If `--train-categories` is used, the script uses `scikit-learn` to cluster uncategorized ("Other") transactions and interactively prompts the user to assign new categories and patterns. These are saved to `categories.yaml`.

## 4. CLI Arguments

The script uses `argparse` to handle command-line inputs:

| Argument | Description |
| :--- | :--- |
| `--input-file` | Path to the input CSV file (required). |
| `--config` | Path to the YAML configuration file (default: `categories.yaml`). |
| `--plot` | If set, displays the generated plot in an interactive window. |
| `--show-matched-categories-only` | If set, prints a sorted list of categorized transactions and exits. |
| `--start` / `-s` | Start date for filtering (format: YYYY-MM-DD). |
| `--end` / `-e` | End date for filtering (format: YYYY-MM-DD). |
| `--train-categories` | Initiates an interactive workflow to categorize "Other" items using clustering. |
| `--filter` | Filters transactions to include only those matching the specified category. |
| `--year` / `-y` | Shortcut to filter for a specific year (e.g. 2023). Mutually exclusive with start/end. |
| `--month` / `-m` | Shortcut to filter for a specific month (e.g. 2023-05). Mutually exclusive with start/end/year. |
| `--show-labels` | Shows value labels on the chart segments (defaults to False). |

## 5. Outputs

The script expects a `reports/` directory to exist (or creates it) and saves:
-   `cleaned_expenses.csv`: The original data with an added `Category` column.
-   `summary.csv`: Aggregated spending totals.
-   `expense_chart.png`: The generated visualization.
