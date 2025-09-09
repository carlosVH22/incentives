import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

st.set_page_config(page_title="Dashboard Incentivos", layout="wide")
st.title("📊 Dashboard de TGMV - Real vs Plan vs YoY")

st.sidebar.header("Carga de Datos")
st.sidebar.markdown("""
**Estructura df actual:**  
`Date,TGMV`  
**Estructura df plan:**  
`Week,TGMV Plan,TGMV 2024`
""")
file_current = st.sidebar.file_uploader("Sube df actual (CSV)", type="csv")
file_plan = st.sidebar.file_uploader("Sube df plan (CSV)", type="csv")

if file_current and file_plan:
    # Cargar y normalizar columnas
    df = pd.read_csv(file_current)
    df_plan = pd.read_csv(file_plan)
    df.columns = df.columns.str.strip().str.lower()
    df_plan.columns = df_plan.columns.str.strip().str.lower()

    # Validar columnas mínimas
    if not {'date','tgmv'}.issubset(df.columns) or not {'week','tgmv plan','tgmv 2024'}.issubset(df_plan.columns):
        st.error("Verifica que los archivos tengan las columnas correctas.")
        st.stop()

    # Columna fecha
    df['date'] = pd.to_datetime(df['date'])

    # Encontrar el primer domingo del año de la PRIMERA fecha presente en df
    year = df['date'].dt.year.min()
    first_sunday = pd.Timestamp(f"{year}-01-01")
    while first_sunday.weekday() != 6:  # 6 es domingo
        first_sunday += pd.Timedelta(days=1)

    # Calcular el número de semana Mercado Libre (domingo-sábado, semana 1 empieza primer domingo)
    df['week'] = ((df['date'] - first_sunday).dt.days // 7) + 1

    st.info("Ejemplo de asignación de semana (Mercado Libre):")
    st.dataframe(df[['date','week']].head(10))

    # Agrupar por semana
    df_weekly = df.groupby('week', as_index=False).agg({'tgmv': 'sum'})
    df_weekly.rename(columns={'tgmv':'real'}, inplace=True)

    # Normalizar nombres y tipos en plan
    df_plan.rename(columns={'tgmv 2024':'yoy','tgmv plan':'plan'}, inplace=True)
    df_plan['week'] = df_plan['week'].astype(int)
    df_weekly['week'] = df_weekly['week'].astype(int)

    # Merge alineando por semana Mercado Libre
    df_final = pd.merge(df_weekly, df_plan, on="week", how='inner')

    # Calcular diferencia YoY y cumplimiento
    df_final["diff_yoy"] = df_final["real"] - df_final["yoy"]
    df_final["cumplimiento"] = np.where(df_final["plan"] != 0, df_final["real"] / df_final["plan"] * 100, np.nan)

    st.subheader("📋 Vista previa de los datos procesados")
    st.dataframe(df_final.head())

    # ---- GRÁFICO 1: Evolución semanal ---
    st.subheader("📈 Evolución semanal: Real vs Plan vs YoY")
    chart1 = (
        alt.Chart(df_final)
        .transform_fold(["real","plan","yoy"], as_=["Métrica","Valor"])
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("Valor:Q", title="TGMV"),
            color="Métrica:N",
            tooltip=["week","Métrica","Valor"]
        ).properties(width=800, height=400)
    )
    st.altair_chart(chart1, use_container_width=True)

    # ---- GRÁFICO 2: Diferencia YoY ---
    st.subheader("📉 Diferencia Real vs YoY")
    chart2 = (
        alt.Chart(df_final)
        .mark_bar()
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("diff_yoy:Q", title="Diferencia YoY"),
            color=alt.condition("datum.diff_yoy > 0", alt.value("green"), alt.value("red")),
            tooltip=["week", "real", "yoy", "diff_yoy"]
        ).properties(width=800, height=400)
    )
    st.altair_chart(chart2, use_container_width=True)

    # ---- GRÁFICO 3: Cumplimiento ---
    st.subheader("📊 Cumplimiento vs Plan")
    chart3 = (
        alt.Chart(df_final)
            .mark_bar()
            .encode(
                x=alt.X("week:O", title="Semana"),
                y=alt.Y("cumplimiento:Q", title="% Cumplimiento Plan"),
                color=alt.condition("datum.cumplimiento >= 100", alt.value("green"), alt.value("orange")),
                tooltip=["week", "real", "plan", "cumplimiento"]
            )
            .properties(width=800, height=400)
    )
    st.altair_chart(chart3, use_container_width=True)

    # Descarga
    st.download_button("Descargar datos procesados", df_final.to_csv(index=False).encode('utf-8'), "datos_dashboard.csv", "text/csv")

    st.success("✅ Dashboard generado con éxito. Pasa el cursor sobre los gráficos para ver los datos.")
else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
