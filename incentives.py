import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Dashboard Incentivos", layout="wide")

st.title("📊 Dashboard de TGMV - Real vs Plan vs YoY")

# --- Subida de archivos ---
st.sidebar.header("Carga de Datos")
file_current = st.sidebar.file_uploader("Sube df actual (CSV)", type="csv")
file_plan = st.sidebar.file_uploader("Sube df plan (CSV)", type="csv")

if file_current and file_plan:
    # Cargar datos
    df = pd.read_csv(file_current)
    df_plan = pd.read_csv(file_plan)

    # Normalizar nombres de columnas
    df.columns = df.columns.str.strip().str.lower()
    df_plan.columns = df_plan.columns.str.strip().str.lower()

    # Procesar df actual → agrupar por semana
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.isocalendar().week

    df_weekly = df.groupby("week", as_index=False).agg({"tgmv": "sum"})
    df_weekly.rename(columns={"tgmv": "real"}, inplace=True)

    # Procesar plan
    df_plan.rename(columns={"tgmv 2024": "yoy", "tgmv plan": "plan"}, inplace=True)

    # Merge por semana
    df_final = pd.merge(df_weekly, df_plan, on="week", how="inner")

    # Calcular diferencia YoY
    df_final["diff_yoy"] = df_final["real"] - df_final["yoy"]

    st.subheader("📋 Vista previa de los datos procesados")
    st.dataframe(df_final.head())

    # --- Gráfico 1: Real vs Plan vs YoY ---
    st.subheader("📈 Evolución semanal: Real vs Plan vs YoY")
    chart1 = (
        alt.Chart(df_final)
        .transform_fold(
            ["real", "plan", "yoy"],
            as_=["Métrica", "Valor"]
        )
        .mark_line(point=True)
        .encode(
            x=alt.X("week:N", title="Semana"),
            y=alt.Y("Valor:Q", title="TGMV"),
            color="Métrica:N",
            tooltip=["week", "Métrica", "Valor"]
        )
        .properties(width=800, height=400)
    )
    st.altair_chart(chart1, use_container_width=True)

    # --- Gráfico 2: Diferencia YoY ---
    st.subheader("📉 Diferencia Real vs YoY")
    chart2 = (
        alt.Chart(df_final)
        .mark_bar()
        .encode(
            x=alt.X("week:N", title="Semana"),
            y=alt.Y("diff_yoy:Q", title="Diferencia YoY"),
            color=alt.condition("datum.diff_yoy > 0", alt.value("green"), alt.value("red")),
            tooltip=["week", "real", "yoy", "diff_yoy"]
        )
        .properties(width=800, height=400)
    )
    st.altair_chart(chart2, use_container_width=True)

    # --- Gráfico 3: Comparación Real vs Plan ---
    st.subheader("📊 Cumplimiento vs Plan")
    df_final["cumplimiento"] = df_final["real"] / df_final["plan"] * 100

    chart3 = (
        alt.Chart(df_final)
        .mark_bar()
        .encode(
            x=alt.X("week:N", title="Semana"),
            y=alt.Y("cumplimiento:Q", title="% Cumplimiento Plan"),
            color=alt.condition("datum.cumplimiento >= 100", alt.value("green"), alt.value("orange")),
            tooltip=["week", "real", "plan", "cumplimiento"]
        )
        .properties(width=800, height=400)
    )
    st.altair_chart(chart3, use_container_width=True)

    st.success("✅ Dashboard generado con éxito. Pasa el cursor sobre los gráficos para ver los datos.")
else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
