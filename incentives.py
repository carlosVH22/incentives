import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df['date'] = pd.to_datetime(df['date'])
    # Calcular semana Mercado Libre (domingo-sábado)
    year = df['date'].dt.year.min()
    first_sunday = pd.Timestamp(f"{year}-01-01")
    while first_sunday.weekday() != 6:
        first_sunday += pd.Timedelta(days=1)
    df['week'] = ((df['date'] - first_sunday).dt.days // 7) + 1

    # Agrupar por semana
    df_weekly = df.groupby('week', as_index=False).agg({'tgmv': 'sum'})
    df_weekly.rename(columns={'tgmv':'real'}, inplace=True)

    # Preparar plan
    df_plan.rename(columns={'tgmv 2024':'yoy','tgmv plan':'plan'}, inplace=True)
    df_plan['week'] = df_plan['week'].astype(int)
    df_weekly['week'] = df_weekly['week'].astype(int)

    # Merge    
    df_final = pd.merge(df_weekly, df_plan, on="week", how='inner')

    # Calcular diferencia YoY
    df_final["diff_yoy"] = df_final["real"] - df_final["yoy"]
    df_final["cumplimiento"] = np.where(df_final["plan"] != 0, 
                                        df_final["real"] / df_final["plan"] * 100, np.nan)

    st.subheader("📋 Vista previa de los datos procesados")
    st.dataframe(df_final.head())

    weeks = df_final['week'].astype(str)

    # --- Gráfico 1: Evolución Real vs Plan vs YoY ---
    st.subheader("📈 Evolución semanal: Real vs Plan vs YoY")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    ax1.plot(weeks, df_final['real'], marker='o', label='Real')
    ax1.plot(weeks, df_final['plan'], marker='o', label='Plan')
    ax1.plot(weeks, df_final['yoy'], marker='o', label='YoY')
    ax1.set_xlabel('Semana')
    ax1.set_ylabel('TGMV')
    ax1.set_title('Evolución semanal: Real vs Plan vs YoY')
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1)

    # --- Gráfico 2: Diferencia YoY ---
    st.subheader("📉 Diferencia Real vs YoY")
    fig2, ax2 = plt.subplots(figsize=(10,4))
    colors = ['green' if val > 0 else 'red' for val in df_final['diff_yoy']]
    ax2.bar(weeks, df_final['diff_yoy'], color=colors)
    ax2.set_xlabel('Semana')
    ax2.set_ylabel('Diferencia YoY')
    ax2.set_title('Diferencia Real vs YoY')
    ax2.axhline(0, color='black', linestyle='--')
    st.pyplot(fig2)

    # --- Gráfico 3: Cumplimiento vs Plan ---
    st.subheader("📊 Cumplimiento vs Plan (%)")
    fig3, ax3 = plt.subplots(figsize=(10,4))
    colormap = ['green' if val >= 100 else 'orange' for val in df_final['cumplimiento']]
    ax3.bar(weeks, df_final['cumplimiento'], color=colormap)
    ax3.set_xlabel('Semana')
    ax3.set_ylabel('% Cumplimiento')
    ax3.set_title('% Cumplimiento vs Plan')
    ax3.axhline(100, color='blue', linestyle='--',label="100% Plan")
    ax3.legend()
    st.pyplot(fig3)

    st.download_button("Descargar datos procesados", df_final.to_csv(index=False).encode('utf-8'),
                       "datos_dashboard.csv", "text/csv")

    st.success("✅ Dashboard generado con éxito. Puedes ver los gráficos arriba. Si necesitas detalles interactivos, puedes usar la tabla vista previa.")
else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
