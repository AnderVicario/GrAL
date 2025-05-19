import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

class ETFAgent:
    """
    ETFAgent: Clase para analizar una empresa y sus ETFs relacionados.
    Estrategias disponibles:
      - Normalización de precios
      - Correlación de retornos
      - Momentum relativo
      - Comparativa de volúmenes
      - Cálculo de residual (diferencia de rendimiento)
      - Detección de crossovers de rendimiento
      - Análisis de regresión empresa~ETF
    """
    def __init__(self, empresa: str, etfs: list, start_date: str = '2023-01-01', end_date: str = None):
        """
        Inicializa el agente con el ticker de la empresa y su lista de ETFs.
        :param empresa: Ticker de la empresa (e.g., 'AAPL').
        :param etfs: Lista de tickers de ETFs relacionados (e.g., ['XLK', 'QQQ']).
        :param start_date: Fecha de inicio para datos históricos.
        :param end_date: Fecha de fin para datos históricos. Si None, usa la fecha actual.
        """
        self.empresa = empresa
        self.etfs = etfs
        self.tickers = [empresa] + etfs
        self.start_date = start_date
        self.end_date = end_date
        self.data = None  # DataFrame con precios y volúmenes
        self.normalized = None  # DataFrame con precios normalizados

    def fetch_data(self):
        """
        Descarga datos históricos (Close y Volume) para la empresa y ETFs.
        """
        df = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False
        )
        close = df['Close']
        volume = df['Volume']
        self.data = pd.concat({'Close': close, 'Volume': volume}, axis=1)
        return self.data

    def normalize_prices(self):
        """
        Normaliza los precios de cierre al mismo punto de partida (base 100).
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de normalizar precios.")
        close = self.data['Close']
        normalized = close.div(close.iloc[0]).mul(100)
        self.normalized = normalized
        return self.normalized

    def compute_correlation(self) -> pd.Series:
        """
        Calcula la correlación de retornos diarios entre la empresa y cada ETF.
        Retorna una Series con la correlación de la empresa frente a cada ETF.
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de calcular correlaciones.")
        returns = self.data['Close'].pct_change().dropna()
        corr_matrix = returns.corr()
        corr_empresa = corr_matrix[self.empresa].drop(labels=[self.empresa])
        return corr_empresa

    def compute_momentum(self, window: int = 30) -> pd.Series:
        """
        Calcula el momentum (rendimiento en los últimos 'window' días) para
        la empresa y los ETFs.
        :return: Series con momentum en porcentaje para cada ticker.
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de calcular momentum.")
        close = self.data['Close']
        past = close.shift(window)
        momentum = (close.iloc[-1] - past.iloc[-1]) / past.iloc[-1] * 100
        return momentum

    def compare_volume(self, window: int = 20) -> pd.DataFrame:
        """
        Calcula estadísticas de volumen (media móvil, z-score) para detectar
        anomalías y compara volúmenes de la empresa vs ETFs.
        :return: DataFrame con columnas ['Volume', 'MA', 'ZScore'] para cada ticker.
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de analizar volumen.")
        volume = self.data['Volume']
        ma = volume.rolling(window).mean()
        std = volume.rolling(window).std()
        zscore = (volume - ma) / std
        result = pd.DataFrame({
            'Volume': volume.iloc[-1],
            'MA': ma.iloc[-1],
            'ZScore': zscore.iloc[-1]
        })
        return result

    def compute_residual(self) -> pd.Series:
        """
        Calcula la diferencia de rendimiento acumulado normalizado entre
        la empresa y cada ETF.
        :return: Series con residuales en base de 100.
        """
        if self.normalized is None:
            raise ValueError("Debe llamar a normalize_prices() antes de calcular residual.")
        norm = self.normalized
        residual = norm[self.empresa] - norm.drop(columns=[self.empresa])
        last_residual = residual.iloc[-1]
        return last_residual

    def detect_crossovers(self) -> pd.DataFrame:
        """
        Identifica los puntos donde el rendimiento normalizado de la empresa
        cruza el rendimiento de los ETFs.
        :return: DataFrame con columnas ['Date', 'ETF', 'Direction'].
        """
        if self.normalized is None:
            raise ValueError("Debe llamar a normalize_prices() antes de detectar crossovers.")
        df = self.normalized
        results = []
        # Para cada ETF, buscar cambios de signo en la diferencia
        for etf in self.etfs:
            diff = df[self.empresa] - df[etf]
            sign = np.sign(diff)
            change = sign != sign.shift(1)
            crosses = df.index[change.fillna(False)]
            for date in crosses:
                direction = 'up' if diff.loc[date] > 0 else 'down'
                results.append({'Date': date, 'ETF': etf, 'Direction': direction})
        return pd.DataFrame(results)

    def regression_analysis(self) -> dict:
        """
        Ajusta un modelo de regresión lineal para predecir los retornos de la empresa
        a partir de los retornos de los ETFs.
        :return: Diccionario con coeficientes, R^2 y residuos.
        """
        if self.data is None:
            raise ValueError("Debe llamar a fetch_data() antes de análisis de regresión.")
        # Preparar retornos
        returns = self.data['Close'].pct_change().dropna()
        y = returns[self.empresa]
        X = returns[self.etfs]
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        return {
            'coefficients': pd.Series(model.coef_, index=self.etfs),
            'intercept': model.intercept_,
            'r2': model.score(X, y),
            'residuals': residuals
        }

    def run_all(self):
        """
        Ejecuta todas las estrategias y devuelve los resultados en un dict.
        """
        results = {}
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