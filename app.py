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
    page_title="Análisis Financiero de Activos",
    page_icon="📊",
    layout="wide"
)


CUSTOM_CSS = """
<style>
:root {
    --bg-main: #020814;
    --bg-sidebar: #071426;
    --bg-panel: #091629;
    --bg-panel-2: #0d1b2f;
    --border: #1f3a56;
    --text-main: #f8fafc;
    --text-muted: #a7b4c5;
    --pink: #ff2f73;
    --cyan: #22c7ff;
    --green: #39ff63;
    --yellow: #ffd21f;
    --orange: #ff8a00;
    --red: #ff1744;
}

.stApp {
    background:
        radial-gradient(circle at 18% 0%, rgba(34, 199, 255, 0.10), transparent 28%),
        radial-gradient(circle at 80% 12%, rgba(255, 47, 115, 0.10), transparent 25%),
        linear-gradient(135deg, var(--bg-main) 0%, #020b18 48%, #000814 100%);
    color: var(--text-main);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-sidebar) 0%, #04101f 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-main);
}

h1, h2, h3 {
    color: var(--text-main);
    letter-spacing: -0.02em;
}

[data-testid="stCaptionContainer"] {
    color: var(--text-muted);
}

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(9, 22, 41, 0.98), rgba(13, 27, 47, 0.96));
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 17px 18px 14px 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.30);
}

div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--green) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 18px;
    border-bottom: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    color: var(--text-main);
    border-radius: 8px 8px 0 0;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: var(--pink) !important;
    border-bottom: 3px solid var(--pink);
}

.stButton > button {
    background: linear-gradient(90deg, #ff1744 0%, var(--pink) 100%);
    color: white;
    border: 0;
    border-radius: 10px;
    font-weight: 800;
    min-height: 44px;
    box-shadow: 0 8px 20px rgba(255, 47, 115, 0.25);
}

.stDownloadButton > button {
    background: transparent;
    color: var(--text-main);
    border: 1px solid var(--pink);
    border-radius: 10px;
    font-weight: 800;
    min-height: 42px;
}

.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #061427 !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
}

.stSlider [data-testid="stTickBar"] {
    background: var(--border);
}

.stSlider [role="slider"] {
    background: var(--pink);
    border-color: var(--pink);
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: var(--border);
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def style_correlation(value: float) -> str:
    if pd.isna(value):
        return "background-color: #091629; color: #a7b4c5"

    abs_value = abs(value)

    if abs_value >= 0.8:
        return "background-color: #e0004d; color: #ffffff; font-weight: 800"
    if 0.4 <= abs_value < 0.8:
        return "background-color: #ff8a00; color: #000000; font-weight: 800"
    return "background-color: #008f39; color: #ffffff; font-weight: 800"


def style_covariance_factory(max_abs_covariance: float):
    def style_covariance(value: float) -> str:
        if pd.isna(value):
            return "background-color: #091629; color: #a7b4c5"

        if max_abs_covariance <= 0 or pd.isna(max_abs_covariance):
            intensity = 0
        else:
            intensity = min(abs(value) / max_abs_covariance, 1)

        if intensity >= 0.70:
            return "background-color: #b00020; color: #ffffff; font-weight: 800"
        if intensity >= 0.35:
            return "background-color: #ff6d00; color: #ffffff; font-weight: 800"
        return "background-color: #ffc400; color: #000000; font-weight: 800"

    return style_covariance


def apply_correlation_style(df: pd.DataFrame):
    styler = df.style.format("{:.4f}", na_rep="N/A")
    try:
        return styler.map(style_correlation)
    except AttributeError:
        return styler.applymap(style_correlation)


def apply_covariance_style(df: pd.DataFrame):
    max_abs_covariance = df.abs().max().max() if not df.empty else 0
    styler = df.style.format("{:.6f}", na_rep="N/A")
    covariance_style = style_covariance_factory(max_abs_covariance)

    try:
        return styler.map(covariance_style)
    except AttributeError:
        return styler.applymap(covariance_style)


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
        correlations_df.to_excel(writer, sheet_name="Correlación")
        covariance_df.to_excel(writer, sheet_name="Covarianza")

    return output.getvalue()


# ============================================================
# Interfaz
# ============================================================

st.title("Análisis Financiero de Activos")
st.caption("Datos de Yahoo Finance vía yfinance")

with st.sidebar:
    st.header("Parámetros de Análisis")

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

    confidence_level_pct = st.number_input(
        "Nivel de confianza para VaR (%)",
        min_value=0.01,
        max_value=99.99,
        value=95.00,
        step=0.01,
        format="%.2f",
        help="Captura cualquier escala porcentual entre 0.01% y 99.99%."
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
        "Indicadores",
        "Índice de referencia",
        "Correlación",
        "Covarianza",
        "Descargar",
    ]
)

with tab_indicators:
    st.subheader("Indicadores por Activo")

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
    st.subheader("Matriz de Correlación (Rendimientos)")

    if correlations_df.empty:
        st.info("Se requieren al menos dos activos válidos con observaciones suficientes para calcular correlaciones.")
    else:
        st.dataframe(apply_correlation_style(correlations_df), use_container_width=True)

with tab_cov:
    st.subheader("Matriz de Covarianza Anualizada (Rendimientos)")

    if covariance_df.empty:
        st.info("Se requieren al menos dos activos válidos con observaciones suficientes para calcular covarianza.")
    else:
        st.dataframe(apply_covariance_style(covariance_df), use_container_width=True)

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
