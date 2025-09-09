import streamlit as st
import pandas as pd
import altair as alt

# Configuración inicial
st.set_page_config(page_title="Dashboard Incentivos", layout="wide")

st.title("📊 Dashboard de Incentivos con comparación YoY y Plan")

# --- Cargar datos ---
st.sidebar.header("Carga de Datos")
current_file = st.sidebar.file_uploader("Sube datos actuales (CSV)", type="csv")
yoy_file = st.sidebar.file_uploader("Sube datos YoY y Plan (CSV)", type="csv")

if current_file is not None and yoy_file is not None:
    # Cargar ambos CSV
    df_current = pd.read_csv(current_file)
    df_yoy = pd.read_csv(yoy_file)

    # Normalizar nombres de columnas
    df_current.columns = df_current.columns.str.strip().str.lower()
    df_yoy.columns = df_yoy.columns.str.strip().str.lower()

    # Supongamos que df_current tiene: ["ciudad", "semana", "real", "prediccion"]
    # y df_yoy tiene: ["ciudad", "semana", "real_año_pasado", "plan", "target"]

    # Unir por ciudad y semana
    df = pd.merge(df_current, df_yoy, on=["ciudad", "semana"], how="inner")

    st.subheader("📋 Vista previa de los datos")
    st.dataframe(df.head())

    # --- Gráfico 1: Evolución Real vs Predicción ---
    st.subheader("📈 Evolución Real vs Predicción")
    chart1 = (
        alt.Chart(df)
        .transform_fold(
            ["real", "prediccion"],
            as_=["Métrica", "Valor"]
        )
        .mark_line(point=True)
        .encode(
            x="semana:N",
            y="Valor:Q",
            color="Métrica:N",
            tooltip=["ciudad", "semana", "Métrica", "Valor"]
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart1, use_container_width=True)

    # --- Gráfico 2: Comparación YoY ---
    st.subheader("📊 Comparación YoY vs Plan")
    chart2 = (
        alt.Chart(df)
        .transform_fold(
            ["real", "real_año_pasado", "plan", "target"],
            as_=["Métrica", "Valor"]
        )
        .mark_line(point=True)
        .encode(
            x="semana:N",
            y="Valor:Q",
            color="Métrica:N",
            tooltip=["ciudad", "semana", "Métrica", "Valor"]
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart2, use_container_width=True)

    # --- Gráfico 3: Diferencia YoY ---
    st.subheader("📉 Diferencia Real vs Año Pasado")
    df["diferencia_yoy"] = df["real"] - df["real_año_pasado"]

    chart3 = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x="semana:N",
            y="diferencia_yoy:Q",
            color=alt.condition("datum.diferencia_yoy > 0", alt.value("green"), alt.value("red")),
            tooltip=["ciudad", "semana", "diferencia_yoy"]
        )
        .properties(width=700, height=400)
    )
    st.altair_chart(chart3, use_container_width=True)

    # --- Extras: Selector de ciudad ---
    st.sidebar.subheader("🔍 Filtros")
    ciudades = st.sidebar.multiselect("Selecciona ciudad", df["ciudad"].unique(), default=df["ciudad"].unique())

    df_filtrado = df[df["ciudad"].isin(ciudades)]

    st.subheader("📌 Datos filtrados por ciudad")
    st.dataframe(df_filtrado)

    st.success("✅ Dashboard generado con éxito. Usa el cursor sobre los gráficos para ver los valores exactos.")

else:
    st.warning("Por favor carga ambos archivos CSV para continuar.")
