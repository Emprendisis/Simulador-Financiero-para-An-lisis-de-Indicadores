import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.stats import norm


# ============================================================
# Configuración general
# ============================================================

st.set_page_config(
    page_title="Análisis Financiero de Activos Bursátiles",
    page_icon="📊",
    layout="wide"
)

PERIOD_MAP = {
    "1 mes": "1mo",
    "2 meses": "2mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "12 meses": "1y",
    "2 años": "2y",
    "3 años": "3y",
    "5 años": "5y",
}

INTERVAL_MAP = {
    "Diaria": "1d",
    "Semanal": "1wk",
    "Mensual": "1mo",
}

ANNUALIZATION_FACTORS = {
    "Diaria": 252,
    "Semanal": 52,
    "Mensual": 12,
}


@dataclass
class AssetMetrics:
    ticker: str
    capital: float
    annual_return: float
    annual_volatility: float
    beta: float
    capm_return: float
    alpha: float
    sharpe: float
    treynor: float
    var_pct: float
    var_abs: float


# ============================================================
# Funciones financieras
# ============================================================

def clean_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def try_mexican_suffix(ticker: str) -> list[str]:
    ticker = clean_ticker(ticker)
    if ticker.endswith(".MX"):
        return [ticker]
    return [ticker, f"{ticker}.MX"]


@st.cache_data(show_spinner=False)
def download_price_data(ticker: str, period: str, interval: str, price_field: str) -> pd.Series:
    """
    Descarga precios desde Yahoo Finance usando yfinance.
    Intenta el ticker directo y, si no existe, intenta con sufijo .MX.
    """
    candidates = try_mexican_suffix(ticker)

    for candidate in candidates:
        try:
            data = yf.download(
                candidate,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            selected_field = price_field
            if selected_field == "Adj Close" and "Adj Close" not in data.columns:
                selected_field = "Close"

            if selected_field not in data.columns:
                continue

            prices = data[selected_field].dropna()
            prices.name = candidate

            if len(prices) >= 3:
                return prices

        except Exception:
            continue

    return pd.Series(dtype=float, name=clean_ticker(ticker))


def calculate_returns(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def calculate_beta(asset_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan

    asset = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]

    benchmark_variance = benchmark.var()
    if benchmark_variance == 0 or pd.isna(benchmark_variance):
        return np.nan

    covariance = asset.cov(benchmark)
    return covariance / benchmark_variance


def calculate_parametric_var(
    mean_return: float,
    volatility: float,
    confidence_level: float
) -> float:
    """
    VaR paramétrico con distribución normal.
    Devuelve VaR porcentual positivo.
    """
    z_score = norm.ppf(1 - confidence_level)
    var_pct = -(mean_return + z_score * volatility)
    return max(var_pct, 0)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def calculate_asset_metrics(
    ticker: str,
    capital: float,
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    annual_factor: int,
    risk_free_rate_annual: float,
    confidence_level: float
) -> AssetMetrics:

    periodic_rf = (1 + risk_free_rate_annual) ** (1 / annual_factor) - 1

    mean_periodic_return = asset_returns.mean()
    periodic_volatility = asset_returns.std()

    annual_return = (1 + mean_periodic_return) ** annual_factor - 1
    annual_volatility = periodic_volatility * np.sqrt(annual_factor)

    beta = calculate_beta(asset_returns, benchmark_returns)

    benchmark_annual_return = (1 + benchmark_returns.mean()) ** annual_factor - 1
    capm_return = risk_free_rate_annual + beta * (benchmark_annual_return - risk_free_rate_annual) if not pd.isna(beta) else np.nan
    alpha = annual_return - capm_return if not pd.isna(capm_return) else np.nan

    sharpe = safe_ratio(annual_return - risk_free_rate_annual, annual_volatility)
    treynor = safe_ratio(annual_return - risk_free_rate_annual, beta)

    var_pct = calculate_parametric_var(
        mean_return=mean_periodic_return,
        volatility=periodic_volatility,
        confidence_level=confidence_level
    )

    var_abs = capital * var_pct

    return AssetMetrics(
        ticker=ticker,
        capital=capital,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        beta=beta,
        capm_return=capm_return,
        alpha=alpha,
        sharpe=sharpe,
        treynor=treynor,
        var_pct=var_pct,
        var_abs=var_abs
    )


def style_correlation_or_covariance(value: float) -> str:
    if pd.isna(value):
        return ""

    abs_value = abs(value)

    if abs_value >= 0.8:
        return "background-color: #f4cccc"
    if 0.4 <= abs_value < 0.8:
        return "background-color: #fff2cc"
    return "background-color: #d9ead3"


def apply_matrix_style(df: pd.DataFrame):
    styler = df.style.format("{:.4f}", na_rep="N/A")

    try:
        return styler.map(style_correlation_or_covariance)
    except AttributeError:
        return styler.applymap(style_correlation_or_covariance)


def format_percent(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def format_number(x):
    if pd.isna(x):
        return "N/A"
    return f"{x:,.4f}"


def format_currency(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


def build_excel_file(
    indicators_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    covariance_df: pd.DataFrame,
    benchmark_df: pd.DataFrame
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        indicators_df.to_excel(writer, index=False, sheet_name="Indicadores")
        benchmark_df.to_excel(writer, index=False, sheet_name="Benchmark")
        correlations_df.to_excel(writer, sheet_name="Correlaciones")
        covariance_df.to_excel(writer, sheet_name="Covarianza")

    return output.getvalue()


# ============================================================
# Interfaz
# ============================================================

st.title("Análisis financiero de activos bursátiles")
st.caption("Aplicación en Streamlit con datos de Yahoo Finance vía yfinance.")

with st.sidebar:
    st.header("Parámetros de entrada")

    number_of_assets = st.number_input(
        "Número de activos a analizar",
        min_value=1,
        max_value=20,
        value=3,
        step=1
    )

    st.subheader("Activos")
    tickers = []
    capitals = []

    for i in range(int(number_of_assets)):
        col_ticker, col_capital = st.columns([1, 1])
        with col_ticker:
            ticker = st.text_input(
                f"Ticker activo {i + 1}",
                value="",
                key=f"ticker_{i}"
            )
        with col_capital:
            capital = st.number_input(
                f"Capital {i + 1}",
                min_value=0.0,
                value=100000.0,
                step=10000.0,
                key=f"capital_{i}"
            )

        tickers.append(clean_ticker(ticker))
        capitals.append(float(capital))

    benchmark_ticker = clean_ticker(
        st.text_input("Ticker del índice de referencia", value="^GSPC")
    )

    period_label = st.selectbox(
        "Periodo de análisis",
        list(PERIOD_MAP.keys()),
        index=4
    )

    frequency_label = st.selectbox(
        "Frecuencia de datos",
        list(INTERVAL_MAP.keys()),
        index=0
    )

    price_field = st.selectbox(
        "Tipo de precio",
        ["Adj Close", "Close"],
        index=0
    )

    confidence_level_pct = st.slider(
        "Nivel de confianza para VaR",
        min_value=90,
        max_value=99,
        value=95,
        step=1
    )

    risk_free_rate_pct = st.number_input(
        "Tasa libre de riesgo anual (%)",
        min_value=-10.0,
        max_value=100.0,
        value=5.0,
        step=0.25
    )

    run_analysis = st.button("Ejecutar análisis", type="primary")


if not run_analysis:
    st.info("Captura los parámetros en la barra lateral y presiona 'Ejecutar análisis'.")
    st.stop()


# ============================================================
# Validaciones iniciales
# ============================================================

valid_input_tickers = [ticker for ticker in tickers if ticker]

if not valid_input_tickers:
    st.error("Debes capturar al menos un ticker válido.")
    st.stop()

if not benchmark_ticker:
    st.error("Debes capturar un ticker válido para el índice de referencia.")
    st.stop()

period = PERIOD_MAP[period_label]
interval = INTERVAL_MAP[frequency_label]
annual_factor = ANNUALIZATION_FACTORS[frequency_label]
confidence_level = confidence_level_pct / 100
risk_free_rate_annual = risk_free_rate_pct / 100


# ============================================================
# Descarga de datos
# ============================================================

with st.spinner("Descargando datos desde Yahoo Finance..."):
    benchmark_prices = download_price_data(
        benchmark_ticker,
        period=period,
        interval=interval,
        price_field=price_field
    )

    if benchmark_prices.empty:
        st.error(
            f"No se pudieron obtener datos válidos para el índice de referencia: {benchmark_ticker}."
        )
        st.stop()

    benchmark_returns = calculate_returns(benchmark_prices)

    if len(benchmark_returns) < 2:
        st.error("El índice de referencia no tiene suficientes observaciones para calcular rendimientos.")
        st.stop()

    asset_prices = {}
    asset_returns = {}
    invalid_tickers = []

    for ticker in valid_input_tickers:
        prices = download_price_data(
            ticker,
            period=period,
            interval=interval,
            price_field=price_field
        )

        if prices.empty or len(prices) < 3:
            invalid_tickers.append(ticker)
            continue

        returns = calculate_returns(prices)

        if returns.empty or len(returns) < 2 or returns.isna().all():
            invalid_tickers.append(ticker)
            continue

        asset_prices[prices.name] = prices
        asset_returns[prices.name] = returns


if invalid_tickers:
    st.warning(
        "Los siguientes tickers no se pudieron procesar o no tienen datos suficientes: "
        + ", ".join(invalid_tickers)
    )

if not asset_returns:
    st.error("No hay activos válidos para analizar.")
    st.stop()


# ============================================================
# Cálculos
# ============================================================

metrics = []

capital_by_input_ticker = {
    clean_ticker(ticker): capital for ticker, capital in zip(tickers, capitals) if clean_ticker(ticker)
}

for resolved_ticker, returns in asset_returns.items():
    base_ticker = resolved_ticker.replace(".MX", "")
    capital = capital_by_input_ticker.get(resolved_ticker, capital_by_input_ticker.get(base_ticker, 0.0))

    metric = calculate_asset_metrics(
        ticker=resolved_ticker,
        capital=capital,
        asset_returns=returns,
        benchmark_returns=benchmark_returns,
        annual_factor=annual_factor,
        risk_free_rate_annual=risk_free_rate_annual,
        confidence_level=confidence_level
    )

    metrics.append(metric)


indicators_df = pd.DataFrame([m.__dict__ for m in metrics])

indicators_df = indicators_df.rename(
    columns={
        "ticker": "Ticker",
        "capital": "Capital invertido",
        "annual_return": "Rendimiento anualizado",
        "annual_volatility": "Volatilidad anualizada",
        "beta": "Beta",
        "capm_return": "Rendimiento esperado CAPM",
        "alpha": "Alfa",
        "sharpe": "Índice de Sharpe",
        "treynor": "Índice de Treynor",
        "var_pct": "VaR porcentual",
        "var_abs": "VaR absoluto",
    }
)

benchmark_annual_return = (1 + benchmark_returns.mean()) ** annual_factor - 1
benchmark_annual_volatility = benchmark_returns.std() * np.sqrt(annual_factor)
benchmark_sharpe = safe_ratio(benchmark_annual_return - risk_free_rate_annual, benchmark_annual_volatility)

benchmark_df = pd.DataFrame(
    {
        "Benchmark": [benchmark_prices.name],
        "Rendimiento anualizado": [benchmark_annual_return],
        "Volatilidad anualizada": [benchmark_annual_volatility],
        "Índice de Sharpe": [benchmark_sharpe],
        "Tasa libre de riesgo anual": [risk_free_rate_annual],
    }
)

returns_df = pd.concat(asset_returns.values(), axis=1).dropna(how="all")
returns_df.columns = list(asset_returns.keys())
returns_df = returns_df.dropna()

if returns_df.shape[1] >= 2 and len(returns_df) >= 2:
    correlations_df = returns_df.corr()
    covariance_df = returns_df.cov() * annual_factor
else:
    correlations_df = pd.DataFrame()
    covariance_df = pd.DataFrame()


# ============================================================
# Resultados
# ============================================================

tab_indicators, tab_benchmark, tab_corr, tab_cov, tab_download = st.tabs(
    [
        "Indicadores por activo",
        "Índice de referencia",
        "Correlaciones",
        "Covarianza",
        "Descarga Excel",
    ]
)

with tab_indicators:
    st.subheader("Indicadores financieros por activo")

    display_df = indicators_df.copy()

    percent_columns = [
        "Rendimiento anualizado",
        "Volatilidad anualizada",
        "Rendimiento esperado CAPM",
        "Alfa",
        "VaR porcentual",
    ]

    numeric_columns = [
        "Beta",
        "Índice de Sharpe",
        "Índice de Treynor",
    ]

    money_columns = [
        "Capital invertido",
        "VaR absoluto",
    ]

    formatted_df = display_df.copy()

    for col in percent_columns:
        formatted_df[col] = formatted_df[col].apply(format_percent)

    for col in numeric_columns:
        formatted_df[col] = formatted_df[col].apply(format_number)

    for col in money_columns:
        formatted_df[col] = formatted_df[col].apply(format_currency)

    st.dataframe(formatted_df, use_container_width=True)

    st.markdown(
        """
        **Notas metodológicas**
        - VaR: paramétrico con distribución normal.
        - Beta: covarianza activo-benchmark dividida entre varianza del benchmark.
        - CAPM: tasa libre de riesgo + beta × prima de mercado.
        - Alfa: rendimiento anualizado observado menos rendimiento esperado CAPM.
        - Sharpe: exceso de rendimiento sobre volatilidad.
        - Treynor: exceso de rendimiento sobre beta.
        """
    )

with tab_benchmark:
    st.subheader("Indicadores del índice de referencia")

    benchmark_display = benchmark_df.copy()
    for col in ["Rendimiento anualizado", "Volatilidad anualizada", "Tasa libre de riesgo anual"]:
        benchmark_display[col] = benchmark_display[col].apply(format_percent)

    benchmark_display["Índice de Sharpe"] = benchmark_display["Índice de Sharpe"].apply(format_number)

    st.dataframe(benchmark_display, use_container_width=True)

with tab_corr:
    st.subheader("Matriz de correlaciones")

    if correlations_df.empty:
        st.info("Se requieren al menos dos activos válidos con observaciones suficientes para calcular correlaciones.")
    else:
        st.dataframe(apply_matrix_style(correlations_df), use_container_width=True)

with tab_cov:
    st.subheader("Matriz de covarianza anualizada")

    if covariance_df.empty:
        st.info("Se requieren al menos dos activos válidos con observaciones suficientes para calcular covarianza.")
    else:
        st.dataframe(apply_matrix_style(covariance_df), use_container_width=True)

with tab_download:
    st.subheader("Descargar resultados")

    excel_bytes = build_excel_file(
        indicators_df=indicators_df,
        correlations_df=correlations_df,
        covariance_df=covariance_df,
        benchmark_df=benchmark_df
    )

    st.download_button(
        label="Descargar archivo Excel",
        data=excel_bytes,
        file_name="analisis_financiero_activos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.success("Análisis completado correctamente.")
