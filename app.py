import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from statistics import NormalDist

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(page_title="Indicadores Financieros", layout="wide")

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stButton>button {width: 100%;}
    .stTextInput>div>input {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

st.title("📊 Simulador Financiero para Análisis de Indicadores")
st.markdown("Analiza rendimiento, volatilidad, VaR y correlaciones entre activos bursátiles.")

PLAZOS = {
    "1 mes": "1mo", "2 meses": "2mo", "3 meses": "3mo", "6 meses": "6mo",
    "12 meses": "12mo", "2 años": "2y", "3 años": "3y", "5 años": "5y"
}

INTERVALOS = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo"}
FACTORES_ANUALES = {"Diaria": 252, "Semanal": 52, "Mensual": 12}


def descargar_precios(ticker: str, periodo: str, intervalo: str) -> pd.Series:
    """Descarga precios de cierre. Devuelve una serie vacía si no hay datos válidos."""
    ticker = ticker.strip().upper()
    if not ticker:
        return pd.Series(dtype="float64")

    try:
        data = yf.Ticker(ticker).history(period=periodo, interval=intervalo, auto_adjust=True)
        if data.empty or "Close" not in data.columns:
            return pd.Series(dtype="float64")

        precios = data["Close"].dropna()
        precios.name = ticker
        return precios
    except Exception:
        return pd.Series(dtype="float64")


def calcular_indicadores(precios: pd.Series, capital: float, frecuencia: str, nivel_confianza: float) -> dict:
    """Calcula rendimiento anualizado, volatilidad anualizada y VaR paramétrico normal."""
    rendimientos = precios.pct_change().dropna()

    if len(rendimientos) < 2:
        raise ValueError("Datos insuficientes")

    factor = FACTORES_ANUALES[frecuencia]
    media_periodica = rendimientos.mean()
    volatilidad_periodica = rendimientos.std(ddof=1)

    rend_anual = media_periodica * factor
    vol_anual = volatilidad_periodica * np.sqrt(factor)

    # VaR paramétrico normal a 1 periodo de frecuencia seleccionada.
    # Se usa volatilidad periódica, no volatilidad anualizada, para evitar sobrestimar VaR de 1 día/semana/mes.
    z_score = abs(NormalDist().inv_cdf(1 - nivel_confianza))
    var_pct = z_score * volatilidad_periodica
    var_absoluto = capital * var_pct

    return {
        "rend_anual": rend_anual,
        "vol_anual": vol_anual,
        "capital": capital,
        "var_absoluto": var_absoluto,
        "var_pct": var_pct,
    }


def formatear_resultado(indicadores: dict) -> list[str]:
    return [
        f"{indicadores['rend_anual']:.1%}",
        f"{indicadores['vol_anual']:.1%}",
        f"${indicadores['capital']:,.2f}",
        f"-${indicadores['var_absoluto']:,.2f}",
        f"-{indicadores['var_pct']:.1%}",
    ]


def estilo_correlacion(valor):
    """Aplica color únicamente cuando el valor es numérico."""
    if pd.isna(valor):
        return ""

    valor_abs = abs(valor)
    if valor_abs >= 0.8:
        return "background-color: #ff6666; color: black"
    if valor_abs >= 0.4:
        return "background-color: #fff176; color: black"
    return "background-color: #81c784; color: black"


with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")

    num_activos = st.number_input("Número de activos a analizar:", min_value=1, value=2, step=1)

    tickers = []
    for i in range(num_activos):
        ticker = st.text_input(f"Ticker #{i + 1}", key=f"ticker_{i}")
        ticker = ticker.strip().upper()
        if ticker:
            tickers.append(ticker)

    capital = st.number_input("Capital invertido por activo ($):", min_value=0.0, value=1000.0, step=100.0)
    indice_referencia = st.text_input("Ticker del índice de referencia (ej. ^IXIC para NASDAQ):", "^IXIC").strip().upper()

    plazo = st.selectbox("Periodo de análisis:", list(PLAZOS.keys()))
    frecuencia = st.selectbox("Frecuencia de datos:", list(INTERVALOS.keys()))
    nivel_confianza = st.slider("Nivel de confianza para VaR", 0.80, 0.99, 0.95, 0.01)

    calcular = st.button("📈 Ejecutar Simulación")

if calcular:
    if not tickers:
        st.warning("Captura al menos un ticker válido.")
        st.stop()

    resultados = {
        "INDICADOR": [
            "Rendimiento Anualizado",
            "Volatilidad Anualizada",
            "Capital",
            "VaR",
            "VaR (%)",
        ]
    }

    precios_validos = {}
    errores = []

    periodo = PLAZOS[plazo]
    intervalo = INTERVALOS[frecuencia]

    for ticker in tickers:
        precios = descargar_precios(ticker, periodo, intervalo)

        if precios.empty or len(precios) < 3:
            resultados[ticker] = ["Sin datos"] * 5
            errores.append(f"{ticker}: Yahoo Finance no devolvió datos suficientes. Revisa el símbolo. Para acciones mexicanas suele requerirse sufijo .MX, por ejemplo BIMBOA.MX.")
            continue

        try:
            indicadores = calcular_indicadores(precios, capital, frecuencia, nivel_confianza)
            resultados[ticker] = formatear_resultado(indicadores)
            precios_validos[ticker] = precios
        except ValueError:
            resultados[ticker] = ["Datos insuficientes"] * 5
            errores.append(f"{ticker}: datos insuficientes para calcular rendimientos.")

    # Índice de referencia
    if indice_referencia:
        precios_idx = descargar_precios(indice_referencia, periodo, intervalo)
        if precios_idx.empty or len(precios_idx) < 3:
            resultados["Índice"] = ["Sin datos"] * 5
            errores.append(f"{indice_referencia}: índice sin datos suficientes.")
        else:
            try:
                indicadores_idx = calcular_indicadores(precios_idx, capital, frecuencia, nivel_confianza)
                resultados["Índice"] = [
                    f"{indicadores_idx['rend_anual']:.1%}",
                    f"{indicadores_idx['vol_anual']:.1%}",
                    "", "", ""
                ]
                precios_validos["Índice"] = precios_idx
            except ValueError:
                resultados["Índice"] = ["Datos insuficientes"] * 5

    df_resultado = pd.DataFrame(resultados).set_index("INDICADOR")

    st.subheader("📋 Resultados del Análisis")
    st.dataframe(df_resultado, use_container_width=True)

    if errores:
        with st.expander("Ver advertencias de datos"):
            for error in errores:
                st.warning(error)

    st.subheader("🔗 Matriz de Correlaciones")

    if len(precios_validos) >= 2:
        df_precios = pd.concat(precios_validos.values(), axis=1, join="inner")
        df_precios.columns = list(precios_validos.keys())
        df_rendimientos = df_precios.pct_change().dropna(how="all")
        df_corr = df_rendimientos.corr()

        # pandas 2.1+ usa Styler.map; versiones antiguas usan Styler.applymap.
        styler = df_corr.style.format("{:.2f}")
        if hasattr(styler, "map"):
            styler = styler.map(estilo_correlacion)
        else:
            styler = styler.applymap(estilo_correlacion)

        st.dataframe(styler, use_container_width=True)
    else:
        df_corr = pd.DataFrame()
        st.info("Se requieren al menos dos series válidas para calcular correlaciones.")

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_resultado.to_excel(writer, sheet_name="Indicadores", index=True)
        if not df_corr.empty:
            df_corr.to_excel(writer, sheet_name="Correlaciones", index=True)
    output.seek(0)

    st.download_button(
        label="📥 Descargar Excel",
        data=output,
        file_name="resultados_indicadores_financieros.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Captura los parámetros y ejecuta la simulación.")
