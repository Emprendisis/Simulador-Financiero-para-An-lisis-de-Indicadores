import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from io import BytesIO
from scipy.stats import norm

# Configuración de página y estilo
st.set_page_config(page_title="Indicadores Financieros", layout="wide")
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            width: 100%;
        }
        .stTextInput>div>input {
            font-size: 1.1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Simulador Financiero para Análisis de Indicadores")
st.markdown("Analiza rendimiento, volatilidad, VaR y correlaciones entre activos bursátiles.")

with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    num_activos = st.number_input("Número de activos a analizar:", min_value=1, step=1)

    tickers = []
    for i in range(num_activos):
        ticker = st.text_input(f"Ticker #{i+1}", key=f"ticker_{i}")
        if ticker:
            tickers.append(ticker.upper())

    capital = st.number_input("Capital invertido por activo ($):", min_value=0.0, value=1000.0, step=100.0)
    nivel_confianza = st.slider("Nivel de confianza para VaR (%):", min_value=90, max_value=99, value=95, step=1)
    indice_referencia = st.text_input("Ticker del índice de referencia (ej. ^IXIC para NASDAQ):", "^IXIC")

    plazo = st.selectbox(
        "Periodo de análisis:",
        ["1 mes", "2 meses", "3 meses", "6 meses", "12 meses", "2 años", "3 años", "5 años"]
    )
    plazos = {
        "1 mes": "1mo",
        "2 meses": "2mo",
        "3 meses": "3mo",
        "6 meses": "6mo",
        "12 meses": "12mo",
        "2 años": "2y",
        "3 años": "3y",
        "5 años": "5y"
    }

    frecuencia = st.selectbox("Frecuencia de datos:", ["Diaria", "Semanal", "Mensual", "Anual"])
    intervalos = {"Diaria": "1d", "Semanal": "1wk", "Mensual": "1mo", "Anual": "3mo"}

    calcular = st.button("📈 Ejecutar Simulación")

if calcular and tickers:
    resultados = {
        "INDICADOR": [
            "Rendimiento Anualizado",
            "Volatilidad Anualizada",
            "Capital",
            "VaR",
            "VaR (%)"
        ]
    }
    precios = {}

    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period=plazos[plazo], interval=intervalos[frecuencia])
            data['Rendimiento'] = data['Close'].pct_change()
            precios[ticker] = data['Close']
            rend_anual = np.mean(data['Rendimiento']) * 252 if frecuencia == 'Diaria' else np.mean(data['Rendimiento']) * 52 if frecuencia == 'Semanal' else np.mean(data['Rendimiento']) * 12
            vol_anual = np.std(data['Rendimiento']) * np.sqrt(252) if frecuencia == 'Diaria' else np.std(data['Rendimiento']) * np.sqrt(52) if frecuencia == 'Semanal' else np.std(data['Rendimiento']) * np.sqrt(12)
            z_score = norm.ppf(nivel_confianza / 100)
            var_absoluto = capital * z_score * vol_anual
            var_pct = var_absoluto / capital

            resultados[ticker] = [
                f"{rend_anual:.1%}",
                f"{vol_anual:.1%}",
                f"${capital:,.2f}",
                f"-${var_absoluto:,.2f}",
                f"-{var_pct:.1%}"
            ]
        except:
            resultados[ticker] = ["Error"] * 5

    try:
        data_idx = yf.Ticker(indice_referencia).history(period=plazos[plazo], interval=intervalos[frecuencia])
        data_idx['Rendimiento'] = data_idx['Close'].pct_change()
        precios["Índice"] = data_idx['Close']
        rend_idx = np.mean(data_idx['Rendimiento']) * 252 if frecuencia == 'Diaria' else np.mean(data_idx['Rendimiento']) * 52 if frecuencia == 'Semanal' else np.mean(data_idx['Rendimiento']) * 12
        vol_idx = np.std(data_idx['Rendimiento']) * np.sqrt(252) if frecuencia == 'Diaria' else np.std(data_idx['Rendimiento']) * np.sqrt(52) if frecuencia == 'Semanal' else np.std(data_idx['Rendimiento']) * np.sqrt(12)

        resultados["Índice"] = [
            f"{rend_idx:.1%}",
            f"{vol_idx:.1%}",
            "",
            "",
            ""
        ]
    except:
        resultados["Índice"] = ["Error"] * 5

    df_resultado = pd.DataFrame(resultados).set_index("INDICADOR")

    st.subheader("📋 Resultados del Análisis")
    st.dataframe(df_resultado, use_container_width=True)

    # Matriz de correlación con formato condicional
    st.subheader("🔗 Matriz de Correlaciones")
    df_corr = pd.DataFrame(precios).pct_change().corr()

    def color_celda(val):
        if val >= 0.8:
            color = 'red'
        elif val >= 0.4:
            color = 'yellow'
        else:
            color = 'lightgreen'
        return f'background-color: {color}; color: black'

    df_corr_formateada = df_corr.style.applymap(color_celda).format("{:.2f}")
    st.dataframe(df_corr_formateada, use_container_width=True)

    # Exportación
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_resultado.to_excel(writer, sheet_name="Indicadores", index=True)
        df_corr.to_excel(writer, sheet_name="Correlaciones", index=True)
    output.seek(0)

    st.download_button(
        label="📥 Descargar Excel",
        data=output,
        file_name="resultados_indicadores_financieros.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
