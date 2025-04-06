import yfinance as yf
import math

def analyze_company_summary(ticker, n_reports=3, frequency="yearly"):
    if not ticker:
        print("Ticker is required.")
        return

    company = yf.Ticker(ticker)
    info = company.info

    # Main Data (profile and ratios) filtered to remove None/NaN values
    main_data = {
        'Name': info.get('longName'),
        'Sector': info.get('sector'),
        'Industry': info.get('industry'),
        'Employees': info.get('fullTimeEmployees'),
        'Headquarters': f"{info.get('city')}, {info.get('country')}",
        'P/E': info.get('trailingPE'),
        'PEG': info.get('pegRatio'),
        'P/B': info.get('priceToBook'),
        'ROE': info.get('returnOnEquity'),
        'ROA': info.get('returnOnAssets'),
        'Debt/Equity': info.get('debtToEquity'),
        'Revenue': info.get('totalRevenue'),
        'Net Income': info.get('netIncome'),
        'EBITDA': info.get('ebitda'),
        'Profit Margin': info.get('profitMargins'),
        'Dividend': info.get('dividendYield')
    }

    filtered_data = {
        k: v for k, v in main_data.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }

    print("=== Main Data ===")
    for key, value in filtered_data.items():
        print(f"{key}: {value}")

    # Helper to print last n reports
    def print_n_reports(label, df, n):
        print(f"\n=== {label} (last {n} reports, {frequency}) ===")
        if df is not None and not df.empty:
            selected = df.iloc[:, :n].dropna(how='all', axis=1)
            if not selected.empty:
                print(selected)
            else:
                print("No valid data")
        else:
            print("No data available")

    # Load and print financials using selected frequency
    print_n_reports("Earnings", company.get_income_stmt(freq=frequency), n_reports)
    print_n_reports("Balance Sheet", company.get_balance_sheet(freq=frequency), n_reports)
    print_n_reports("Cash Flow", company.get_cash_flow(freq=frequency), n_reports)

# Example usage
analyze_company_summary("IBE.MC", n_reports=3, frequency="quarterly")
