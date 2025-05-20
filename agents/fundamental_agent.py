import yfinance as yf
import math
import pandas as pd
from datetime import datetime

class FundamentalAnalysisAgent:
    def __init__(
        self,
        company: str,
        ticker: str,
        sector: str,
        start_date: str = None,
        end_date: str = None
    ):
        self.company = company
        self.ticker = ticker
        self.sector = sector

        # parse or default our dates
        try:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
            self.end_date   = datetime.strptime(end_date,   "%Y-%m-%d") if end_date   else None
        except ValueError:
            # malformed dates → drop to None
            self.start_date = None
            self.end_date   = None

        # derive report frequency & how many past periods to fetch
        self.frequency, self.n_reports = self._parse_date_span()

        # containers for fetched data
        self.profile_data    = {}
        self.income_statement = None
        self.balance_sheet    = None
        self.cash_flow        = None

    def _parse_date_span(self):
        """
        Decide frequency ('quarterly' or 'yearly') and number of past reports
        based on the span between start_date and end_date:
        
        - ≤ 30 days    → quarterly, 2 periods
        - 31-120 days  → quarterly, 3 periods
        - 121-364 days → quarterly, 4 periods
        - 365-730 days → yearly,    3 periods
        - > 730 days   → yearly,    5 periods
        
        If either date is missing or invalid, default to yearly, 3.
        """
        if not self.start_date or not self.end_date:
            return "yearly", 3

        span_days = (self.end_date - self.start_date).days
        if span_days <= 30:
            return "quarterly", 2
        if span_days <= 120:
            return "quarterly", 3
        if span_days < 365:
            return "quarterly", 4
        if span_days < 730:
            return "yearly", 3
        return "yearly", 5

    
    def fetch_company_data(self):
        try:
            company = yf.Ticker(self.ticker)
            info = company.info
            
            # Eskuratu enpresaren profilaren datuak
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
            
            # Iragazi None/NaN balioak
            self.profile_data = {
                k: v for k, v in self.profile_data.items()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            }
            
            # Eskuratu finantza-egoerak
            self.income_statement = company.get_income_stmt(freq=self.frequency)
            self.balance_sheet = company.get_balance_sheet(freq=self.frequency)
            self.cash_flow = company.get_cash_flow(freq=self.frequency)
            
            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False
    
    def _format_financial_data(self, df, n_reports):
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Eskuratu azken n txostenak
        selected = df.iloc[:, :n_reports].dropna(how='all', axis=1)
        if selected.empty:
            return pd.DataFrame()
        
        # Formateatu zenbakizko balioak
        for col in selected.columns:
            selected[col] = selected[col].apply(
                lambda x: f"{x/1e6:.2f}M" if isinstance(x, (int, float)) else x
            )
        
        return selected
    
    def calculate_key_metrics(self):
        metrics = {}
        
        # Beharrezko datuak baditugu bakarrik kalkulatu metrikak
        if self.income_statement is not None and not self.income_statement.empty:
            # Get the most recent period
            recent = self.income_statement.iloc[:, 0]
            
            if 'TotalRevenue' in recent and 'NetIncome' in recent:
                net_income = recent['NetIncome']
                revenue = recent['TotalRevenue']
                if isinstance(net_income, (int, float)) and isinstance(revenue, (int, float)):
                    metrics['Net Profit Margin'] = net_income / revenue if revenue != 0 else None
            
            # Urtetik urterako hazkundea, gutxienez 2 aldi baditugu
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
        
        # Iragazi None/NaN balioak
        metrics = {
            k: v for k, v in metrics.items()
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        }
        
        return metrics
    
    def process(self):
        if not self.fetch_company_data():
            return {"status": "error", "message": f"Failed to fetch data for {self.ticker}"}
        
        # Formateatu irteerako finantza-egoerak
        income_stmt = self._format_financial_data(self.income_statement, self.n_reports)
        balance_sheet = self._format_financial_data(self.balance_sheet, self.n_reports)
        cash_flow = self._format_financial_data(self.cash_flow, self.n_reports)
        
        # Metrika gehigarriak kalkulatu
        additional_metrics = self.calculate_key_metrics()
        
        # Analisi-txosten osoa sortu
        analysis_report = {
            "company_info": self.profile_data,
            "analysis_parameters": {
                "start_date": self.start_date.strftime("%Y-%m-%d") if self.start_date else None,
                "end_date": self.end_date.strftime("%Y-%m-%d") if self.end_date else None,
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