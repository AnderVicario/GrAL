class TechnicalAnalysisAgent:
    def __init__(self, company, ticker, date_range, resolution="daily"):
        self.company = company
        self.ticker = ticker
        self.date_range = date_range
        self.resolution = resolution

    def process(self):
        return {
            "report_type": "TechnicalAnalysis",
            "company": self.company,
            "ticker": self.ticker,
            "indicators": {
                "RSI": 55,
                "MACD": "Señal alcista"
            }
        }
