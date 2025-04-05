import yfinance as yf
import math
import pandas as pd

class FundamentalAnalysisAgent:
    def __init__(self, company, ticker, sector, date_range):
        self.company = company
        self.ticker = ticker
        self.sector = sector
        self.date_range = date_range
        # Parse date_range to determine frequency and number of reports
        self.frequency, self.n_reports = self._parse_date_range()
        # Initialize data containers
        self.profile_data = {}
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
        
    def _parse_date_range(self):
        """Parse date_range string to determine frequency and number of reports."""
        if not self.date_range:
            return "yearly", 3  # Default values
        
        # Extract number and unit (h for hours, d for days)
        try:
            if 'h' in self.date_range:
                value = int(self.date_range.replace('h', ''))
                # For small time frames, use quarterly reports
                return "quarterly", min(12, max(1, value // 3))
            elif 'd' in self.date_range:
                value = int(self.date_range.replace('d', ''))
                # For longer time frames, determine if yearly or quarterly is more appropriate
                if value >= 365:
                    return "yearly", min(10, max(1, value // 365))
                else:
                    return "quarterly", min(12, max(1, value // 90))
            else:
                # Default to yearly if no unit is specified
                value = int(self.date_range)
                return "yearly", min(10, max(1, value))
        except (ValueError, TypeError):
            return "yearly", 3  # Default values if parsing fails
    
    def fetch_company_data(self):
        """Fetch all relevant company data from yfinance."""
        try:
            company = yf.Ticker(self.ticker)
            info = company.info
            
            # Fetch company profile data
            self.profile_data = {
                'Name': info.get('longName', self.company),
                'Ticker': self.ticker,
                'Sector': info.get('sector', self.sector),
                'Industry': info.get('industry'),
                'Employees': info.get('fullTimeEmployees'),
                'Headquarters': f"{info.get('city', 'N/A')}, {info.get('country', 'N/A')}",
                'P/E': info.get('trailingPE'),
                'Forward P/E': info.get('forwardPE'),
                'PEG': info.get('pegRatio'),
                'P/B': info.get('priceToBook'),
                'ROE': info.get('returnOnEquity'),
                'ROA': info.get('returnOnAssets'),
                'Debt/Equity': info.get('debtToEquity'),
                'Revenue': info.get('totalRevenue'),
                'Net Income': info.get('netIncome'),
                'EBITDA': info.get('ebitda'),
                'Profit Margin': info.get('profitMargins'),
                'Operating Margin': info.get('operatingMargins'),
                'Dividend Yield': info.get('dividendYield'),
                'Market Cap': info.get('marketCap'),
                'Beta': info.get('beta'),
                '52-Week High': info.get('fiftyTwoWeekHigh'),
                '52-Week Low': info.get('fiftyTwoWeekLow')
            }
            
            # Filter out None/NaN values
            self.profile_data = {
                k: v for k, v in self.profile_data.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            }
            
            # Fetch financial statements
            self.income_statement = company.get_income_stmt(freq=self.frequency)
            self.balance_sheet = company.get_balance_sheet(freq=self.frequency)
            self.cash_flow = company.get_cash_flow(freq=self.frequency)
            
            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False
    
    def _format_financial_data(self, df, n_reports):
        """Format financial dataframe for reporting."""
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Get the last n reports
        selected = df.iloc[:, :n_reports].dropna(how='all', axis=1)
        if selected.empty:
            return pd.DataFrame()
        
        # Format numeric values
        for col in selected.columns:
            selected[col] = selected[col].apply(
                lambda x: f"{x/1e6:.2f}M" if isinstance(x, (int, float)) else x
            )
        
        return selected
    
    def calculate_key_metrics(self):
        """Calculate additional financial metrics based on available data."""
        metrics = {}
        
        # Calculate metrics only if we have the necessary data
        if self.income_statement is not None and not self.income_statement.empty:
            # Get the most recent period
            recent = self.income_statement.iloc[:, 0]
            
            if 'TotalRevenue' in recent and 'NetIncome' in recent:
                net_income = recent['NetIncome']
                revenue = recent['TotalRevenue']
                if isinstance(net_income, (int, float)) and isinstance(revenue, (int, float)):
                    metrics['Net Profit Margin'] = net_income / revenue if revenue != 0 else None
            
            # Year-over-year growth if we have at least 2 periods
            if self.income_statement.shape[1] >= 2:
                prev_revenue = self.income_statement.loc['TotalRevenue', self.income_statement.columns[1]]
                curr_revenue = self.income_statement.loc['TotalRevenue', self.income_statement.columns[0]]
                
                if isinstance(prev_revenue, (int, float)) and isinstance(curr_revenue, (int, float)) and prev_revenue != 0:
                    metrics['Revenue Growth'] = (curr_revenue - prev_revenue) / prev_revenue
        
        if self.balance_sheet is not None and not self.balance_sheet.empty:
            recent_bs = self.balance_sheet.iloc[:, 0]
            
            if 'TotalAssets' in recent_bs and 'TotalLiab' in recent_bs:
                assets = recent_bs['TotalAssets']
                liabilities = recent_bs['TotalLiab']
                
                if isinstance(assets, (int, float)) and isinstance(liabilities, (int, float)):
                    metrics['Debt Ratio'] = liabilities / assets if assets != 0 else None
        
        # Filter out None/NaN values
        metrics = {
            k: v for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }
        
        return metrics
    
    def process(self):
        """Process all company data and return a comprehensive analysis report."""
        if not self.fetch_company_data():
            return {"status": "error", "message": f"Failed to fetch data for {self.ticker}"}
        
        # Format financial statements for output
        income_stmt = self._format_financial_data(self.income_statement, self.n_reports)
        balance_sheet = self._format_financial_data(self.balance_sheet, self.n_reports)
        cash_flow = self._format_financial_data(self.cash_flow, self.n_reports)
        
        # Calculate additional metrics
        additional_metrics = self.calculate_key_metrics()
        
        # Prepare the full analysis report
        analysis_report = {
            "company_info": self.profile_data,
            "analysis_parameters": {
                "date_range": self.date_range,
                "frequency": self.frequency,
                "reports_analyzed": self.n_reports
            },
            "financial_statements": {
                "income_statement": income_stmt.to_dict() if not income_stmt.empty else {},
                "balance_sheet": balance_sheet.to_dict() if not balance_sheet.empty else {},
                "cash_flow": cash_flow.to_dict() if not cash_flow.empty else {}
            },
            "additional_metrics": additional_metrics
        }
        
        return analysis_report


# Example usage:
def main():
    # Create an instance with appropriate parameters
    agent = FundamentalAnalysisAgent(
        company="Iberdrola", 
        ticker="IBE.MC", 
        sector="Utilities", 
        date_range="300d"  # Will use yearly analysis with appropriate n_reports
    )
    
    # Process the data
    analysis = agent.process()
    
    # Print results in a readable format
    print(f"\n=== {analysis['company_info'].get('Name', 'Company')} ({analysis['company_info'].get('Ticker', 'N/A')}) Analysis ===")
    print(f"Sector: {analysis['company_info'].get('Sector', 'N/A')}")
    print(f"Analysis: {analysis['analysis_parameters']['frequency']} data for {analysis['analysis_parameters']['date_range']}")
    
    print("\n=== Company Profile ===")
    for key, value in analysis['company_info'].items():
        print(f"{key}: {value}")
        
    print("\n=== Key Financial Metrics ===")
    for key, value in analysis['additional_metrics'].items():
        if isinstance(value, float):
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value}")
    
    print("\n=== Income Statement Highlights ===")
    if analysis['financial_statements']['income_statement']:
        income_df = pd.DataFrame(analysis['financial_statements']['income_statement'])
        print(income_df.loc[['TotalRevenue', 'GrossProfit', 'OperatingIncome', 'NetIncome']])
    else:
        print("No income statement data available")
    
    # Similar output for balance sheet and cash flow could be added
    
if __name__ == "__main__":
    main()