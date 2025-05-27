import json
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from together import Together


class ETFAgent:
    """
    ETFAgent: Entitate bat eta harekin erlazionatutako ETF-ak aztertzeko klasea, eta RAG sistemarako chunk-ak itzultzen ditu.
    Epearen arabera egokitutako estrategia hauek erabiltzen dira (laburra, ertaina, luzea):
      - Prezioen normalizazioa
      - Errendimenduen korrelazioa
      - Momentu erlatiboa
      - Bolumen konparazioa
      - Errendimenduaren aldea (errestoa)
      - Errendimenduen gurutzaketak detektatzea
      - Entitatea~ETF erregresio-analisia
    """

    def __init__(self, name: str, ticker: str, sector: str,
                 start_date: str = '2023-01-01', end_date: str = None):
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.name = name
        self.ticker = ticker
        self.sector = sector
        self.etfs = self._identify_etfs()
        self.etfs = [t for t in self.etfs if self.is_valid_ticker(t)]
        self.tickers = [ticker] + self.etfs
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.data = None
        self.normalized = None
        # Epea zehaztu
        delta_days = (self.end_date - self.start_date).days
        self.term = 'short' if delta_days <= 30 else 'medium' if delta_days <= 180 else 'long'

    def _identify_etfs(self):
        prompt = f"""
        You are a financial analyst assistant. Given a financial entity name, its stock ticker and sector, return a list of ETF tickers that are related to this entity. Relationships can be based on sector, industry, country of origin, or inclusion in ETF holdings.

        Relevant ETFs typically correlate with:
        - Publicly traded companies, especially those included in ETF holdings
        - Entities in well-defined sectors (e.g., tech, finance, energy) covered by sector ETFs
        - Companies operating in countries or regions represented by geographic ETFs
        - Assets with ETF exposure (e.g., gold → GLD, Bitcoin → IBIT, US bonds → TLT)
        - Businesses whose performance follows economic cycles reflected in thematic ETFs
        
        Do not return any ETFs if the entity:
        - Is private, not publicly traded, or a startup
        - Operates in an unrepresented niche or illiquid market
        - Is a small-cap cryptocurrency without ETF tracking
        - Is an NGO, government program, or legal structure with no financial market presence
        - Is highly idiosyncratic or has risks not captured by ETFs

        These are some example ETFs that may be relevant include:

        - Sector ETFs: XLF (Financials), XLY (Consumer Discretionary), XLU (Utilities), XLE (Energy), XLI (Industrials), XLC (Communications), XLK (Technology), XLRE (Real Estate), XLB (Materials), XLP (Consumer Staples), XLV (Health Care)
        - Country ETFs: SPY (USA), EWG (Germany), KSA (Saudi Arabia), EWA (Australia), EWZ (Brazil), EWC (Canada), FXI (China), EWY (South Korea), EWP (Spain), EWQ (France), INDA (India), EWI (Italy), EWJ (Japan), EWW (Mexico), EWU (UK), EWS (Singapore), EWL (Switzerland), EWT (Taiwan), TUR (Turkey)
        - Entity type ETFs: GLD (Gold), CWB (Convertible Bonds), PFF (Preferred Shares), HYG (High Yield Bonds), EEM (Emerging Markets), EFA (Developed Markets ex-North America), TIP (Treasury Inflation-Protected Securities), LQD (Investment Grade Bonds), DBC (Commodities), TLT (Long-Term Treasuries)

        Please return a list of ETF tickers that are relevant to the given entity. You may use any of the above ETFs if applicable, but also include others that are appropriate.

        Input format:
        Entity: {self.name}
        Ticker: {self.ticker}
        Sector: {self.sector}

        Output format:
        A plain list of ETF tickers (e.g., SPY, QQQ, XLF). Do not include any explanations, descriptions, or additional text—just the ticker symbols.
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=2056,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<｜end▁of▁sentence｜>"],
            stream=True
        )
        full_response = ""
        for token in response:
            if hasattr(token, 'choices'):
                content = token.choices[0].delta.content
                full_response += content
        cleaned = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
        tickers = re.split(r"[,\n]", cleaned)
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        return tickers

    def fetch_data(self, min_days: int = 30):
        if self.term == 'short':
            self.start_date = self.end_date - timedelta(days=min_days)

        # Descargar los datos
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )

        # Crear DataFrame con Close y Volume
        data = pd.concat({'Close': df['Close'], 'Volume': df['Volume']}, axis=1)

        # Detectar y eliminar columnas que son completamente NaN
        invalid_cols = data.columns[data.isna().all()].tolist()
        if invalid_cols:
            data = data.drop(columns=invalid_cols)
            invalid_tickers = [col[1] for col in invalid_cols if len(col) > 1]  # Extraer ticker del MultiIndex
            if invalid_tickers:
                for ticker in invalid_tickers:
                    if ticker in self.etfs:
                        self.etfs.remove(ticker)
                    if ticker in self.tickers:
                        self.tickers.remove(ticker)

        self.data = data

    def normalize_prices(self):
        # Prezioak normalizatu lehen balio erabilgarriarekin oinarrituta
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")

        close = self.data['Close']

        # Aurkitu lehen balio ez-NaN bakoitzeko (per column)
        first_valid = close.apply(lambda col: col[col.first_valid_index()], axis=0)

        # Normalizatu datuak horrekin
        self.normalized = close.divide(first_valid).multiply(100)
        return self.normalized

    def compute_correlation(self):
        # Errendimenduen arteko korrelazioa kalkulatu
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")
        returns = self.data['Close'].ffill().pct_change().dropna()
        if self.term == 'short':
            returns = returns.tail(30)
        elif self.term == 'medium':
            returns = returns.tail(180)
        corr = returns.corr()[self.ticker].drop(self.ticker)
        return corr

    def compute_momentum(self):
        """
        Momentuma kalkulatu (ehunekotan errendimendua) epearen arabera:
          - laburra: 30 egun arte
          - ertaina: 180 egun arte
          - luzea: datu guztien arabera
        Eskuragarri dauden datuak baino gehiago eskatzen bada, historiko osoa erabiliko da.
        """
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")

        close = self.data['Close']
        desired_days = 30 if self.term == 'short' else 180 if self.term == 'medium' else (
                self.end_date - self.start_date).days
        available_days = len(close) - 1
        window = min(desired_days, available_days)

        if window <= 0:
            return pd.Series({t: np.nan for t in self.tickers}, name="momentum")

        # Azken 'window + 1' egunak hartu
        data_window = close.iloc[-(window + 1):]

        # Aktibo bakoitzerako, lehen eta azken balio baliagarriak hartu (NaN ez direnak)
        start_price = data_window.apply(lambda col: col[col.first_valid_index()], axis=0)
        end_price = data_window.apply(lambda col: col[col.last_valid_index()], axis=0)

        # Momentum kalkulatu: azken prezioa eta hasierakoa erabiliz
        # Biak baliagarriak badira soilik kalkulatu
        momentum = ((end_price - start_price) / start_price * 100).where(start_price.notna() & end_price.notna())

        return momentum

    def compare_volume(self):
        if self.data is None:
            raise ValueError("Daturik ez. Erabili fetch_data() lehenik.")

        window = 5 if self.term == 'short' else 20 if self.term == 'medium' else 60

        if len(self.data) < window:
            raise ValueError(f"Ez dago nahikoa datu. Beharrezkoak: {window}, dauzkagu: {len(self.data)}")

        # Bolumenarekin, NaN balioak betetzea
        vol = self.data['Volume'].ffill().bfill()

        # Kalkuluak egiten dira betetako balioekin
        ma = vol.rolling(window, min_periods=1).mean()  # min_periods=1 gehitu
        std = vol.rolling(window, min_periods=1).std()  # min_periods=1 gehitu

        z = (vol - ma) / std

        return pd.DataFrame({
            'Volume': vol.iloc[-1],
            'MA': ma.iloc[-1],
            'ZScore': z.iloc[-1]
        })

    def compute_residual(self):
        # Entitatearen eta ETF-en arteko errestoak kalkulatu
        if self.normalized is None:
            raise ValueError("No data. Use normalize_prices() first.")
        norm = self.normalized.dropna()
        emp = norm[self.ticker]
        etfs = norm[self.etfs]
        diff = emp.values.reshape(-1, 1) - etfs.values
        df = pd.DataFrame(diff, index=norm.index, columns=self.etfs)
        return df.mean() if self.term == 'long' else df.iloc[-1]

    def detect_crossovers(self):
        # Errendimenduen arteko gurutzaketak detektatu
        if self.normalized is None:
            raise ValueError("No data. Use normalize_prices() first.")
        df = self.normalized.dropna()
        if self.term == 'short':
            df = df.tail(30)
        elif self.term == 'medium':
            df = df.tail(180)
        rows = []
        for etf in self.etfs:
            diff = df[self.ticker] - df[etf]
            sign = np.sign(diff)
            for date in df.index[sign.ne(sign.shift()).fillna(False)]:
                rows.append({'Date': date.isoformat(), 'ETF': etf,
                             'Direction': 'up' if diff.loc[date] > 0 else 'down'})
        return pd.DataFrame(rows)

    def regression_analysis(self):
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")
        ret = self.data['Close'].ffill().pct_change().dropna()
        if self.term == 'short':
            ret = ret.tail(30)
        elif self.term == 'medium':
            ret = ret.tail(180)
        y = ret[self.ticker]
        X = ret[self.etfs]
        m = LinearRegression().fit(X, y)
        pred = m.predict(X)
        resid = y - pred
        return {
            'coefficients': dict(zip(self.etfs, m.coef_)),
            'intercept': float(m.intercept_),
            'r2': float(m.score(X, y)),
            'residuals_summary': resid.describe().to_dict()
        }

    def to_json(self, obj):
        # Objektuak (DataFrame, Series, etab.) JSON bihurtu
        if isinstance(obj, pd.DataFrame): return obj.reset_index().to_dict('records')
        if isinstance(obj, pd.Series): return obj.to_dict()
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, dict): return {k: self.to_json(v) for k, v in obj.items()}
        return obj

    def is_valid_ticker(self, ticker: str) -> bool:
        try:
            info = yf.Ticker(ticker).info
            return info and 'regularMarketPrice' in info
        except Exception:
            return False

    def run_and_chunk(self, base_metadata: dict, max_chars: int = 1500) -> list:
        """
        Analisiak exekutatu eta JSON chunk-en zerrenda itzuli.
        Chunk bakoitza dict bat da: testua (JSON) eta metadatuak.
        """
        if self.tickers.__len__() <= 1:
            return None

        if self.data is None:
            self.fetch_data()
        if self.normalized is None:
            self.normalize_prices()

        outputs = {
            'correlation': self.compute_correlation(),
            'momentum': self.compute_momentum(),
            'volume': self.compare_volume(),
            'residual': self.compute_residual(),
            'crossovers': self.detect_crossovers(),
            'regression': self.regression_analysis()
        }
        chunks = []
        for atype, result in outputs.items():
            payload = {'analysis_type': atype,
                       'entity': self.ticker,
                       'term': self.term,
                       'result': self.to_json(result)}
            text = json.dumps(payload, ensure_ascii=False)
            # chunk bakarrean sartzen bada
            if len(text) <= max_chars:
                meta = {**base_metadata, 'analysis_type': atype, 'source': 'ETFAgent'}
                chunks.append({'text': text, 'metadata': meta})
            else:
                # testua zatitu max_chars arabera
                for i in range(0, len(text), max_chars):
                    part = text[i:i + max_chars]
                    meta = {**base_metadata, 'analysis_type': atype, 'chunk_number': i // max_chars + 1,
                            'total_chunks': len(text) // max_chars + 1, 'source': 'ETFAgent'}
                    chunks.append({'text': part, 'metadata': meta})
        return chunks

# if __name__ == '__main__':
#     # Adibideko parametroak
#     from dotenv import load_dotenv
#     load_dotenv()
#     base_metadata = {
#         'entity': 'Apple',
#         'ticker': 'AAPL',
#         'entity_type': 'stock',
#         'report_date': pd.Timestamp.today().isoformat(),
#         'expiration_date': None
#     }
#     # Instantziatu eta exekutatu
#     # agent = ETFAgent(name='EURO', ticker='EURUSD=X', sector='currency', start_date='2025-03-01', end_date='2025-05-19')
#     agent = ETFAgent(name='Bitcoin', ticker='BTC-USD', sector='crypto', start_date='2025-05-17', end_date='2025-05-19')
#
#     chunks = agent.run_and_chunk(base_metadata=base_metadata, max_chars=1000)
#     print(chunks)
