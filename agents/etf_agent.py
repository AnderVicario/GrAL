import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class ETFAgent:
    """
    ETFAgent: Clase para analizar una empresa y sus ETFs relacionados.
    Estrategias disponibles con adaptación según plazo (corto, medio, largo):
      - Normalización de precios
      - Correlación de retornos
      - Momentum relativo
      - Comparativa de volúmenes
      - Cálculo de residual (diferencia de rendimiento)
      - Detección de crossovers de rendimiento
      - Análisis de regresión empresa~ETF
    """
    def __init__(self, empresa: str, etfs: list,
                 start_date: str = '2023-01-01', end_date: str = None):
        self.empresa = empresa
        self.etfs = etfs
        self.tickers = [empresa] + etfs
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.data = None
        self.normalized = None
        # Determinar plazo según rango de fechas
        delta_days = (self.end_date - self.start_date).days
        if delta_days <= 30:
            self.term = 'corto'
        elif delta_days <= 180:
            self.term = 'medio'
        else:
            self.term = 'largo'

    def fetch_data(self):
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date.strftime('%Y-%m-%d'),
            end=self.end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=False
        )
        close = df['Close']
        volume = df['Volume']
        self.data = pd.concat({'Close': close, 'Volume': volume}, axis=1)
        return self.data

    def normalize_prices(self):
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de normalizar precios.")
        close = self.data['Close']
        self.normalized = close.div(close.iloc[0]).mul(100)
        return self.normalized

    def compute_correlation(self) -> pd.Series:
        """
        Calcula la correlación de retornos diarios en un subperiodo según plazo:
          - corto: últimos 30 días
          - medio: últimos 180 días
          - largo: todo el periodo
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de calcular correlaciones.")
        returns = self.data['Close'].pct_change().dropna()
        if self.term == 'corto':
            returns = returns.tail(30)
        elif self.term == 'medio':
            returns = returns.tail(180)
        corr_matrix = returns.corr()
        return corr_matrix[self.empresa].drop(labels=[self.empresa])

    def compute_momentum(self) -> pd.Series:
        """
        Momentum según plazo:
          - corto: últimos días hasta start, max 30 días
          - medio: últimos 180 días
          - largo: desde start
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de calcular momentum.")
        close = self.data['Close']
        if self.term == 'corto':
            window = min((self.end_date - self.start_date).days, 30)
        elif self.term == 'medio':
            window = min((self.end_date - self.start_date).days, 180)
        else:
            window = (self.end_date - self.start_date).days
        past = close.shift(window)
        return (close.iloc[-1] - past.iloc[-1]) / past.iloc[-1] * 100

    def compare_volume(self) -> pd.DataFrame:
        """
        Análisis de volumen con ventana según plazo:
          - corto: MA y z-score en 5 días
          - medio: en 20 días
          - largo: en 60 días
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de analizar volumen.")
        volume = self.data['Volume']
        if self.term == 'corto':
            window = 5
        elif self.term == 'medio':
            window = 20
        else:
            window = 60
        ma = volume.rolling(window).mean()
        std = volume.rolling(window).std()
        zscore = (volume - ma) / std
        return pd.DataFrame({
            'Volume': volume.iloc[-1],
            'MA': ma.iloc[-1],
            'ZScore': zscore.iloc[-1]
        })

    def compute_residual(self) -> pd.Series:
        """
        Residual entre empresa y ETFs en el periodo según plazo:
          - corto/medio: diferencia normalizada al final
          - largo: promedio de residuales de todo el periodo
        """
        if self.normalized is None:
            raise ValueError("Debe llamar a normalize_prices() antes de calcular residual.")
        norm = self.normalized
        emp = norm[self.empresa]
        etf_df = norm.drop(columns=[self.empresa])
        diff = emp.subtract(etf_df, axis=0)
        if self.term == 'largo':
            # devolver estadístico resumen
            return diff.mean()
        else:
            return diff.iloc[-1]

    def detect_crossovers(self) -> pd.DataFrame:
        """
        Detecta cruces en un subperiodo según plazo:
          - corto: últimos 30 días
          - medio: 180 días
          - largo: todo el periodo
        """
        if self.normalized is None:
            raise ValueError("Debe llamar a normalize_prices() antes de detectar crossovers.")
        df = self.normalized
        if self.term == 'corto':
            df = df.tail(30)
        elif self.term == 'medio':
            df = df.tail(180)
        results = []
        for etf in self.etfs:
            diff = df[self.empresa] - df[etf]
            sign = np.sign(diff)
            change = sign != sign.shift(1)
            for date in df.index[change.fillna(False)]:
                direction = 'up' if diff.loc[date] > 0 else 'down'
                results.append({'Date': date, 'ETF': etf, 'Direction': direction})
        return pd.DataFrame(results)

    def regression_analysis(self) -> dict:
        """
        Regresión sobre retornos en subperiodo según plazo:
          - corto: últimos 30 días
          - medio: 180 días
          - largo: todo el periodo
        Devuelve coeficientes, R2 e indicadores de residuales.
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de análisis de regresión.")
        returns = self.data['Close'].pct_change().dropna()
        if self.term == 'corto':
            returns = returns.tail(30)
        elif self.term == 'medio':
            returns = returns.tail(180)
        y = returns[self.empresa]
        X = returns[self.etfs]
        model = LinearRegression().fit(X, y)
        preds = model.predict(X)
        resid = y - preds
        return {
            'coefficients': pd.Series(model.coef_, index=self.etfs),
            'intercept': model.intercept_,
            'r2': model.score(X, y),
            'residuals_summary': resid.describe()
        }

    def run_all(self) -> dict:
        results = {'term': self.term}
        results['data'] = self.fetch_data()
        results['normalized'] = self.normalize_prices()
        results['correlation'] = self.compute_correlation()
        results['momentum'] = self.compute_momentum()
        results['volume_analysis'] = self.compare_volume()
        results['residual'] = self.compute_residual()
        results['crossovers'] = self.detect_crossovers()
        results['regression'] = self.regression_analysis()
        return results

if __name__ == '__main__':
    agente = ETFAgent(empresa='AAPL', etfs=['XLK', 'QQQ', 'SPY'], start_date='2023-01-01')
    resultados = agente.run_all()
    print("Resultados de análisis:")
    print(resultados['correlation'])
    print("Resultados de análisis:")
    print(resultados['momentum'])
    print("Resultados de análisis:")
    print(resultados['volume_analysis'])
    print("Resultados de análisis:")
    print(resultados['residual'])