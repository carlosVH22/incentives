import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet

st.set_page_config(page_title="Dashboard Incentivos", layout="wide")
st.title("📊 Dashboard de TGMV: Real vs Plan vs Predicción (semanas custom)")

st.sidebar.header("Carga los archivos CSV")
plan_file = st.sidebar.file_uploader("Sube el plan diario 2025 (date, plan)", type="csv")
real_file = st.sidebar.file_uploader("Sube los datos reales (date, tgmv), debe contener 2024 y 2025", type="csv")

# --- FUNCION DE ASIGNACION DE SEMANAS CUSTOM ---
def assign_custom_weeks(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    year = df['date'].dt.year.iloc[0]
    all_days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    start = all_days[0]
    while start.weekday() != 5 and start <= all_days[-1]:
        start += pd.Timedelta(days=1)
    week_ends = [start]
    while week_ends[-1] + pd.Timedelta(days=1) <= all_days[-1]:
        next_end = week_ends[-1] + pd.Timedelta(days=7)
        if next_end > all_days[-1]:
            next_end = all_days[-1]
        week_ends.append(next_end)
    week_id = 1
    curr_start = all_days[0]
    for w_end in week_ends:
        mask = (df['date'] >= curr_start) & (df['date'] <= w_end)
        df.loc[mask, 'week'] = week_id
        df.loc[mask, 'week_start'] = curr_start
        df.loc[mask, 'week_end'] = w_end
        curr_start = w_end + pd.Timedelta(days=1)
        week_id += 1
    df['week'] = df['week'].astype(int)
    return df

if plan_file and real_file:
    # --- PLAN ---
    df_plan = pd.read_csv(plan_file)
    df_plan.columns = df_plan.columns.str.strip().str.lower()
    df_plan['date'] = pd.to_datetime(df_plan['date'])
    df_plan = assign_custom_weeks(df_plan)
    plan_weekly = df_plan.groupby(['week','week_start','week_end'])['plan'].sum().reset_index()

    # --- REAL ---
    df_real = pd.read_csv(real_file)
    df_real.columns = df_real.columns.str.strip().str.lower()
    df_real['date'] = pd.to_datetime(df_real['date'])

    real_2025 = df_real[df_real['date'].dt.year == 2025]
    real_2025 = assign_custom_weeks(real_2025)
    real_2025w = real_2025.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
    real_2025w.rename(columns={'tgmv':'real'}, inplace=True)

    real_2024 = df_real[df_real['date'].dt.year == 2024]
    if len(real_2024) > 0:
        real_2024 = assign_custom_weeks(real_2024)
        real_2024w = real_2024.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
        real_2024w.rename(columns={'tgmv':'yoy'}, inplace=True)
    else:
        real_2024w = pd.DataFrame(columns=['week','week_start','week_end','yoy'])

    df_weeks = plan_weekly.merge(real_2025w, on=['week','week_start','week_end'], how='left')
    df_weeks = df_weeks.merge(real_2024w[['week','yoy']], on='week', how='left')

    # --- PREDICCIÓN ---
    st.sidebar.subheader("⚡ Predicción con Prophet")
    periods = st.sidebar.slider("Semanas a predecir", 1, 20, 6)

    if len(real_2025) > 0:
        df_prophet = real_2025[['date','tgmv']].rename(columns={'date':'ds','tgmv':'y'})
        model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        model.fit(df_prophet)

        future = model.make_future_dataframe(periods=periods*7, freq='D')
        forecast = model.predict(future)

        forecast['date'] = forecast['ds']
        forecast = assign_custom_weeks(forecast)
        forecast_weekly = forecast.groupby(['week','week_start','week_end'])[['yhat','yhat_lower','yhat_upper']].sum().reset_index()

        df_weeks = df_weeks.merge(forecast_weekly, on=['week','week_start','week_end'], how='outer')

    # --- Labels ---
    df_weeks['semana_lbl'] = df_weeks.apply(
        lambda x: f"Semana {x['week']} ({x['week_start'].date()} a {x['week_end'].date()})" if pd.notnull(x['week_start']) else f"Semana {x['week']}",
        axis=1
    )

    # Identificar semanas futuras (sin datos reales)
    df_weeks['is_future'] = df_weeks['real'].isna()

    # --- VISTA PREVIA ---
    st.subheader("📋 Vista previa")
    st.dataframe(df_weeks[['week','week_start','week_end','plan','real','yoy','yhat','diff_yoy','cumplimiento']].head(20))

    # --- 📈 GRÁFICO 1: Plan vs Real vs Predicción ---
    st.subheader("📈 Real vs Plan vs Predicción")
    melted = df_weeks.melt(
        id_vars=['week','semana_lbl','is_future'],
        value_vars=['real','plan','yhat'],
        var_name='Métrica', value_name='Valor'
    )

    chart_real = alt.Chart(melted[melted['Métrica'].isin(['real','plan'])]).mark_line(point=True).encode(
        x="week:O",
        y="Valor:Q",
        color=alt.Color('Métrica:N'),
        tooltip=['week','semana_lbl','Métrica','Valor']
    )

    chart_pred = alt.Chart(melted[melted['Métrica']=='yhat']).mark_line(point=True, strokeDash=[5,5], color="blue").encode(
        x="week:O",
        y="Valor:Q",
        tooltip=['week','semana_lbl','Métrica','Valor']
    )

    band = alt.Chart(df_weeks[df_weeks['is_future']==True]).mark_area(opacity=0.2, color="lightblue").encode(
        x="week:O",
        y="yhat_lower:Q",
        y2="yhat_upper:Q"
    )

    st.altair_chart(chart_real + chart_pred + band, use_container_width=True)

    # --- 📉 GRÁFICO 2: Diferencia Real vs YoY / Predicción ---
    st.subheader("📉 Diferencia vs YoY / Predicción")
    df_weeks['diff_display'] = np.where(df_weeks['is_future'],
                                        df_weeks['yhat'] - df_weeks['yoy'],
                                        df_weeks['real'] - df_weeks['yoy'])

    chart2 = alt.Chart(df_weeks).mark_bar().encode(
        x="week:O",
        y="diff_display:Q",
        color=alt.condition(
            "datum.is_future",
            alt.value("lightgreen"),  # predicción
            alt.condition("datum.diff_display > 0", alt.value("green"), alt.value("red"))
        ),
        tooltip=['week','semana_lbl','real','yoy','yhat','diff_display']
    )
    st.altair_chart(chart2, use_container_width=True)

    # --- 📊 GRÁFICO 3: Cumplimiento vs Plan ---
    st.subheader("📊 Cumplimiento vs Plan (%)")
    df_weeks['cumpl_display'] = np.where(df_weeks['is_future'],
                                         df_weeks['yhat']/df_weeks['plan']*100,
                                         df_weeks['real']/df_weeks['plan']*100)

    chart3 = alt.Chart(df_weeks).mark_bar().encode(
        x="week:O",
        y="cumpl_display:Q",
        color=alt.condition(
            "datum.is_future",
            alt.value("lightblue"),
            alt.condition("datum.cumpl_display >= 100", alt.value("green"), alt.value("orange"))
        ),
        tooltip=['week','semana_lbl','real','plan','yhat','cumpl_display']
    )
    st.altair_chart(chart3, use_container_width=True)

else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
