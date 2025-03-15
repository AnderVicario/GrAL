import yfinance as yf
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

tickers = ['AMZN', 'MSFT', 'GOOGL', 'NVDA', 'CRM', 'PLTR']

class YahooAPI:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_csv = f"all_tsx_ticker_info_{self.timestamp}.csv"

    def fetch_ticker_info(self, symbol):  # Agregar `self`
        stock = yf.Ticker(symbol)
        try:
            info = stock.info
            info['ticker'] = symbol
            return info
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def fetch_all_tickers(self, symbols, max_workers=5):
        print("Fetching data for tickers...")
        data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(self.fetch_ticker_info, symbols)  # Aquí ya no hay error
            for result in results:
                if result:
                    data.append(result)
        return data
    
    def save_to_csv(self, data):
        print(f"Saving data to {self.output_csv}...")
        df = pd.DataFrame(data)
        df.to_csv(self.output_csv, index=False, encoding="utf-8")
        print(f"Data successfully saved to {self.output_csv}")

# Ejecutar el código
yahoo = YahooAPI()
ticker_data = yahoo.fetch_all_tickers(tickers)
yahoo.save_to_csv(ticker_data)
print(ticker_data)

