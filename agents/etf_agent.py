import yfinance as yf
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LinearRegression

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
    def __init__(self, entity: str, etfs: list,
                 start_date: str = '2023-01-01', end_date: str = None):
        self.entity = entity
        self.etfs = etfs
        self.tickers = [entity] + etfs
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.data = None
        self.normalized = None
        # Epea zehaztu
        delta_days = (self.end_date - self.start_date).days
        self.term = 'corto' if delta_days <= 30 else 'medio' if delta_days <= 180 else 'largo'

    def fetch_data(self):
        # Finantza-datuak deskargatu (prezioak eta bolumenak)
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
        self.data = pd.concat({ 'Close': df['Close'], 'Volume': df['Volume']}, axis=1)
        return self.data

    def normalize_prices(self):
        # Prezioak normalizatu lehen egunean oinarrituta
        if self.data is None:
            raise ValueError("fetch_data() erabili normalize_prices() baino lehen.")
        close = self.data['Close']
        self.normalized = close.div(close.iloc[0]).mul(100)
        return self.normalized

    def compute_correlation(self):
        # Errendimenduen arteko korrelazioa kalkulatu
        if self.data is None:
            raise ValueError("fetch_data() erabili compute_correlation() baino lehen.")
        returns = self.data['Close'].pct_change().dropna()
        if self.term == 'corto': returns = returns.tail(30)
        elif self.term == 'medio': returns = returns.tail(180)
        corr = returns.corr()[self.entity].drop(self.entity)
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
            raise ValueError("fetch_data() erabili compute_momentum() baino lehen.")
        close = self.data['Close']
        desired_days = 30 if self.term == 'corto' else 180 if self.term == 'medio' else (self.end_date - self.start_date).days
        available_days = len(close) - 1
        window = min(desired_days, available_days)
        if window <= 0:
            raise ValueError("Ez dago nahikoa datu momentum kalkulatzeko.")
        start_price = close.iloc[-window-1]
        end_price = close.iloc[-1]
        return (end_price - start_price) / start_price * 100

    def compare_volume(self):
        # Bolumenaren batezbesteko mugikorra eta ZScore kalkulatu
        if self.data is None:
            raise ValueError("fetch_data() erabili compare_volume() baino lehen.")
        window = 5 if self.term=='corto' else 20 if self.term=='medio' else 60
        vol = self.data['Volume']
        ma = vol.rolling(window).mean()
        z = (vol - ma) / vol.rolling(window).std()
        return pd.DataFrame({ 'Volume': vol.iloc[-1], 'MA': ma.iloc[-1], 'ZScore': z.iloc[-1] })

    def compute_residual(self):
        # Entitatearen eta ETF-en arteko errestoak kalkulatu
        if self.normalized is None:
            raise ValueError("normalize_prices() erabili compute_residual() baino lehen.")
        norm = self.normalized.dropna()
        emp = norm[self.entity]
        etfs = norm[self.etfs]
        diff = emp.values.reshape(-1,1) - etfs.values
        df = pd.DataFrame(diff, index=norm.index, columns=self.etfs)
        return df.mean() if self.term=='largo' else df.iloc[-1]

    def detect_crossovers(self):
        # Errendimenduen arteko gurutzaketak detektatu
        if self.normalized is None:
            raise ValueError("normalize_prices() erabili detect_crossovers() baino lehen.")
        df = self.normalized.dropna()
        if self.term=='corto': df = df.tail(30)
        elif self.term=='medio': df = df.tail(180)
        rows=[]
        for etf in self.etfs:
            diff = df[self.entity] - df[etf]
            sign = np.sign(diff)
            for date in df.index[sign.ne(sign.shift()).fillna(False)]:
                rows.append({'Date': date.isoformat(), 'ETF': etf,
                             'Direction':'up' if diff.loc[date]>0 else 'down'})
        return pd.DataFrame(rows)

    def regression_analysis(self):
        # Erregresio linealaren bidez entitatea~ETF erlazioa aztertu
        if self.data is None:
            raise ValueError("fetch_data() erabili regression_analysis() baino lehen.")
        ret = self.data['Close'].pct_change().dropna()
        if self.term=='corto': ret = ret.tail(30)
        elif self.term=='medio': ret = ret.tail(180)
        y = ret[self.entity]; X = ret[self.etfs]
        m=LinearRegression().fit(X,y); pred=m.predict(X)
        resid=y-pred
        return {
            'coefficients':dict(zip(self.etfs, m.coef_)),
            'intercept':float(m.intercept_),
            'r2':float(m.score(X,y)),
            'residuals_summary':resid.describe().to_dict()
        }

    def to_json(self, obj):
        # Objektuak (DataFrame, Series, etab.) JSON bihurtu
        if isinstance(obj,pd.DataFrame): return obj.reset_index().to_dict('records')
        if isinstance(obj,pd.Series): return obj.to_dict()
        if isinstance(obj,(np.floating, float)): return float(obj)
        if isinstance(obj,(np.integer, int)): return int(obj)
        if isinstance(obj,dict): return {k:self.to_json(v) for k,v in obj.items()}
        return obj

    def run_and_chunk(self, base_metadata: dict, max_chars: int=1500) -> list:
        """
        Analisiak exekutatu eta JSON chunk-en zerrenda itzuli.
        Chunk bakoitza dict bat da: testua (JSON) eta metadatuak.
        """
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
        chunks=[]
        for atype, result in outputs.items():
            payload = {'analysis_type': atype,
                       'entity': self.entity,
                       'term': self.term,
                       'result': self.to_json(result)}
            text = json.dumps(payload, ensure_ascii=False)
            # chunk bakarrean sartzen bada
            if len(text)<=max_chars:
                meta = {**base_metadata, 'analysis_type': atype}
                chunks.append({'text':text,'metadata':meta})
            else:
                # testua zatitu max_chars arabera
                for i in range(0,len(text),max_chars):
                    part=text[i:i+max_chars]
                    meta={**base_metadata,'analysis_type':atype,'chunk_number':i//max_chars+1,'total_chunks':len(text)//max_chars+1,'source':'ETFAgent'}
                    chunks.append({'text':part,'metadata':meta})
        return chunks

if __name__ == '__main__':
    # Adibideko parametroak
    base_metadata = {
        'entity': 'AAPL',
        'ticker': 'AAPL',
        'entity_type': 'stock',
        'report_date': pd.Timestamp.today().isoformat(),
        'expiration_date': None
    }
    # Instantzitu eta exekutatu
    agent = ETFAgent(entity='AAPL', etfs=['XLK', 'QQQ', 'SPY'], start_date='2025-03-01', end_date='2025-05-19')
    chunks = agent.run_and_chunk(base_metadata=base_metadata, max_chars=1000)
    print(chunks)
