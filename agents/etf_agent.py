import json
import re
from datetime import timedelta
from typing import Union

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from together import Together


def _is_valid_ticker(ticker: str) -> bool:
    """
    Egiaztatu ea ticker-ak benetako prezio-datuak dituen.
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        return not hist.empty
    except Exception:
        return False


class ETFAgent:
    """
    ETFAgent klasea entitate finantzario bat eta harekin erlazionatutako ETF-en analisia egiteko.
    Analisiak hiru denbora-epetan egiten dira (laburra, ertaina, luzea) eta emaitzak RAG sistemarako 
    chunk-etan itzultzen dira.
    """

    def __init__(self, name: str, ticker: str, sector: str, start_date: str = '2023-01-01', end_date: str = None):
        """
        ETFAgent-aren hasieratzailea.
        
        Args:
            name: Entitatearen izena
            ticker: Burtsako sinboloa
            sector: Sektorea
            start_date: Hasiera data (lehenetsia: '2023-01-01')
            end_date: Amaiera data (lehenetsia: gaur)
        """
        self.llm_client = Together()
        self.model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.name = name
        self.ticker = ticker
        self.sector = sector
        self.etfs = self._identify_etfs()
        print("All etfs: " + str(self.etfs))
        self.etfs = [t for t in self.etfs if _is_valid_ticker(t)]
        print("Valid etfs: " + str(self.etfs))
        self.tickers = [ticker] + self.etfs
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.data = None
        self.normalized = None
        # Epea zehaztu
        delta_days = (self.end_date - self.start_date).days
        self.term = 'short' if delta_days <= 30 else 'medium' if delta_days <= 180 else 'long'

    def _identify_etfs(self) -> list:
        """
        Entitatearekin erlazionatutako ETF-ak identifikatzen ditu LLM eredua erabiliz.
        
        Returns:
            list: Erlazionatutako ETF-en ticker zerrenda
        """
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

    def _fetch_data(self, min_days: int = 30):
        """
        Yahoo Finance-tik prezio eta bolumen datuak eskuratzen ditu.
        NaN balioak dituzten ETF-ak ezabatzen ditu.
        
        Args:
            min_days: Gutxieneko egun kopurua epe laburrerako
        """
        if self.term == 'short':
            self.start_date = self.end_date - timedelta(days=min_days)

        # Datuak deskargatu
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )

        # DataFrame sortu Close eta Volume-rekin
        data = pd.concat({'Close': df['Close'], 'Volume': df['Volume']}, axis=1)

        # Guztiz NaN diren zutabeak detektatu eta ezabatu
        invalid_cols = data.columns[data.isna().all()].tolist()
        if invalid_cols:
            data = data.drop(columns=invalid_cols)
            invalid_tickers = [col[1] for col in invalid_cols if len(col) > 1]  # Atera tickerra MultiIndexetik
            if invalid_tickers:
                for ticker in invalid_tickers:
                    if ticker in self.etfs:
                        self.etfs.remove(ticker)
                    if ticker in self.tickers:
                        self.tickers.remove(ticker)

        self.data = data

    def _normalize_prices(self) -> pd.DataFrame:
        """
        Prezioak normalizatzen ditu (100eko oinarria) konparaketak errazteko.
        
        Returns:
            DataFrame normalizatutako prezioekin
        """
        # Prezioak normalizatu lehen balio erabilgarriarekin oinarrituta
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")

        close = self.data['Close']

        # Aurkitu lehen balio ez-NaN bakoitzeko (per column)
        first_valid = close.apply(lambda col: col[col.first_valid_index()], axis=0)

        # Normalizatu datuak horrekin
        self.normalized = close.divide(first_valid).multiply(100)
        return self.normalized

    def _compute_correlation(self) -> pd.Series:
        """
        Entitatearen eta ETF-en arteko korrelazio koefizienteak kalkulatzen ditu.
        
        Returns:
            Series ETF bakoitzarekiko korrelazio balioekin
        """
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

    def _compute_momentum(self) -> pd.Series:
        """
        Momentu adierazlea kalkulatzen du epearen arabera:
        - Laburra: 30 egun
        - Ertaina: 180 egun
        - Luzea: Epe osoa
        
        Returns:
            Series ETF bakoitzaren momentu balioekin
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

    def _compare_volume(self) -> pd.DataFrame:
        """
        Bolumenaren analisia egiten du, batezbesteko mugikorra eta Z-score estatistikoak erabiliz.
        
        Returns:
            DataFrame bolumen, batezbesteko mugikor eta Z-score balioekin
        """
        if self.data is None:
            raise ValueError("No data. Use fetch_data() first.")

        window = 5 if self.term == 'short' else 20 if self.term == 'medium' else 60

        if len(self.data) < window:
            raise ValueError(f"Not enough data. Required: {window}, have: {len(self.data)}")

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

    def _compute_residual(self) -> pd.Series:
        """
        Entitatearen eta ETF-en arteko prezio diferentziak kalkulatzen ditu.
        
        Returns:
            Series ETF bakoitzarekiko diferentzia balioekin
        """
        # Entitatearen eta ETF-en arteko errestoak kalkulatu
        if self.normalized is None:
            raise ValueError("No data. Use normalize_prices() first.")
        norm = self.normalized.dropna()
        emp = norm[self.ticker]
        etfs = norm[self.etfs]
        diff = emp.values.reshape(-1, 1) - etfs.values
        df = pd.DataFrame(diff, index=norm.index, columns=self.etfs)
        return df.mean() if self.term == 'long' else df.iloc[-1]

    def _detect_crossovers(self) -> pd.DataFrame:
        """
        Prezioen gurutzaketak detektatzen ditu entitatearen eta ETF-en artean.
        
        Returns:
            DataFrame gurutzaketen data, ETF eta norabidearekin
        """
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

    def _regression_analysis(self) -> dict:
        """
        Erregresio lineala burutzen du entitatearen eta ETF-en artean.
        
        Returns:
            dict erregresioaren emaitza estatistikoekin
        """
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

    def _to_json(self, obj) -> Union[dict, list, float, int]:
        """
        Pandas eta Numpy objektuak JSON formatura bihurtzen ditu.
        
        Args:
            obj: Bihurtu beharreko objektua
            
        Returns:
            JSON bateragarria den objektua
        """
        # Objektuak (DataFrame, Series, etab.) JSON bihurtu
        if isinstance(obj, pd.DataFrame): return obj.reset_index().to_dict('records')
        if isinstance(obj, pd.Series): return obj.to_dict()
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, dict): return {k: self._to_json(v) for k, v in obj.items()}
        return obj

    def run_and_chunk(self, base_metadata: dict, max_chars: int = 1500) -> list:
        """
        Analisi guztiak exekutatu eta emaitzak JSON chunk-etan itzultzen ditu.
        
        Args:
            base_metadata: Oinarrizko metadatuak chunk guztietarako
            max_chars: Chunk bakoitzaren gehienezko karaktere kopurua
            
        Returns:
            list: Chunk-en zerrenda, bakoitza testua eta metadatuekin
        """
        if self.tickers.__len__() <= 1:
            return []

        if not _is_valid_ticker(self.ticker):
            return []

        if self.data is None:
            self._fetch_data()
        if self.normalized is None:
            self._normalize_prices()

        outputs = {
            'correlation': self._compute_correlation(),
            'momentum': self._compute_momentum(),
            'volume': self._compare_volume(),
            'residual': self._compute_residual(),
            'crossovers': self._detect_crossovers(),
            'regression': self._regression_analysis()
        }
        chunks = []
        for atype, result in outputs.items():
            payload = {'analysis_type': atype,
                       'entity': self.ticker,
                       'term': self.term,
                       'result': self._to_json(result)}
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

#
# if __name__ == '__main__':
#     print(_is_valid_ticker('TU=F'))
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
