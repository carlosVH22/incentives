import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet

st.set_page_config(page_title="Dashboard Incentivos", layout="wide")
st.title("📊 Dashboard de TGMV: Real vs Plan vs YoY (semanas custom + predicción)")

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
    # --- LEE Y PROCESA EL PLAN DIARIO 2025 ---
    df_plan = pd.read_csv(plan_file)
    df_plan.columns = df_plan.columns.str.strip().str.lower()
    df_plan['date'] = pd.to_datetime(df_plan['date'])
    df_plan = assign_custom_weeks(df_plan)
    plan_weekly = df_plan.groupby(['week', 'week_start', 'week_end'])['plan'].sum().reset_index()

    # --- LEE Y PROCESA LOS DATOS DIARIOS REALES DE 2024 Y 2025 ---
    df_real = pd.read_csv(real_file)
    df_real.columns = df_real.columns.str.strip().str.lower()
    df_real['date'] = pd.to_datetime(df_real['date'])

    # --- REAL 2025 ---
    real_2025 = df_real[df_real['date'].dt.year == 2025]
    real_2025 = assign_custom_weeks(real_2025)
    real_2025w = real_2025.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
    real_2025w.rename(columns={'tgmv':'real'}, inplace=True)

    # --- REAL 2024 para YoY ---
    real_2024 = df_real[df_real['date'].dt.year == 2024]
    if len(real_2024) > 0:
        real_2024 = assign_custom_weeks(real_2024)
        real_2024w = real_2024.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
        real_2024w.rename(columns={'tgmv':'yoy'}, inplace=True)
    else:
        real_2024w = pd.DataFrame(columns=['week','week_start','week_end','yoy'])

    # --- JUNTAR TODO ---
    df_weeks = plan_weekly.merge(real_2025w, on=['week','week_start','week_end'], how='left')
    df_weeks = df_weeks.merge(real_2024w[['week','yoy']], on='week', how='left')

    df_weeks['diff_yoy'] = df_weeks['real'] - df_weeks['yoy']
    df_weeks['cumplimiento'] = df_weeks['real'] / df_weeks['plan'] * 100

    # Etiquetas tooltip
    df_weeks['semana_lbl'] = df_weeks.apply(
        lambda x: f"Semana {x['week']} ({x['week_start'].date()} a {x['week_end'].date()})", axis=1
    )

    # --- 📈 PREDICCIÓN CON PROPHET ---
    st.sidebar.subheader("⚡ Predicción con Prophet")
    periods = st.sidebar.slider("Semanas a predecir", 1, 12, 6)

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

    # --- Vista previa ---
    st.subheader("📋 Vista previa de las semanas agregadas")
    st.dataframe(df_weeks[['week','week_start','week_end','plan','real','yoy','yhat','yhat_lower','yhat_upper','diff_yoy','cumplimiento']].head(15))

    # --- Gráfico 1: Real vs Plan vs YoY + Forecast ---
    st.subheader("📈 Evolución semanal con predicción")
    melted = df_weeks.melt(
        id_vars=['week','semana_lbl'],
        value_vars=['real','plan','yoy','yhat'],
        var_name='Métrica', value_name='Valor'
    )

    selection = alt.selection_point(fields=['week'])
    chart1 = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("Valor:Q", title="TGMV"),
            color=alt.Color('Métrica:N', legend=alt.Legend(title="Métrica")),
            tooltip=[
                alt.Tooltip('week:O', title='Semana'),
                alt.Tooltip('semana_lbl', title='Rango de fechas'),
                alt.Tooltip('Métrica:N'),
                alt.Tooltip('Valor:Q', format=',')
            ],
            opacity=alt.condition(selection, alt.value(1), alt.value(0.7)),
        )
        .add_params(selection)
        .interactive()
        .properties(height=400, width=850)
    )

    # Banda de incertidumbre Prophet
    band = (
        alt.Chart(df_weeks)
        .mark_area(opacity=0.2, color="lightblue")
        .encode(
            x="week:O",
            y="yhat_lower:Q",
            y2="yhat_upper:Q"
        )
    )

    st.altair_chart(band + chart1, use_container_width=True)

    # --- Gráfico 2 y 3 (sin cambios) ---
    st.subheader("📉 Diferencia Real vs YoY (interactivo)")
    selection2 = alt.selection_point(fields=['week'])
    chart2 = (
        alt.Chart(df_weeks)
        .mark_bar()
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("diff_yoy:Q", title="Diferencia YoY"),
            color=alt.condition("datum.diff_yoy > 0", alt.value("green"), alt.value("red")),
            tooltip=["week","semana_lbl","real","yoy","diff_yoy"],
            opacity=alt.condition(selection2, alt.value(1), alt.value(0.7))
        )
        .add_params(selection2)
        .interactive()
        .properties(height=350, width=850)
    )
    st.altair_chart(chart2, use_container_width=True)

    st.subheader("📊 Cumplimiento vs Plan (%) (interactivo)")
    selection3 = alt.selection_point(fields=['week'])
    chart3 = (
        alt.Chart(df_weeks)
        .mark_bar()
        .encode(
            x="week:O",
            y="cumplimiento:Q",
            color=alt.condition("datum.cumplimiento >= 100", alt.value("green"), alt.value("orange")),
            tooltip=["week","semana_lbl","real","plan","cumplimiento"],
            opacity=alt.condition(selection3, alt.value(1), alt.value(0.7))
        )
        .add_params(selection3)
        .interactive()
        .properties(height=350, width=850)
    )
    st.altair_chart(chart3, use_container_width=True)

else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
