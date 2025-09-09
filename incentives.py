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
n_pred_weeks = st.sidebar.slider("Semanas a predecir", 2, 20, 6)

# --- FUNCION DE ASIGNACION DE SEMANAS CUSTOM ---
def assign_custom_weeks(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    year = df['date'].dt.year.iloc[0]
    all_days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    start = all_days[0]
    while start.weekday() != 5 and start <= all_days[-1]:  # primer sábado
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
    # --- LEE Y PROCESA EL PLAN ---
    df_plan = pd.read_csv(plan_file)
    df_plan.columns = df_plan.columns.str.strip().str.lower()
    df_plan['date'] = pd.to_datetime(df_plan['date'])
    df_plan = assign_custom_weeks(df_plan)
    plan_weekly = df_plan.groupby(['week','week_start','week_end'])['plan'].sum().reset_index()

    # --- LEE Y PROCESA LOS DATOS REALES ---
    df_real = pd.read_csv(real_file)
    df_real.columns = df_real.columns.str.strip().str.lower()
    df_real['date'] = pd.to_datetime(df_real['date'])

    # Real 2025
    real_2025 = df_real[df_real['date'].dt.year == 2025]
    real_2025 = assign_custom_weeks(real_2025)
    real_2025w = real_2025.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
    real_2025w.rename(columns={'tgmv':'real'}, inplace=True)

    # Real 2024 para YoY
    real_2024 = df_real[df_real['date'].dt.year == 2024]
    if len(real_2024) > 0:
        real_2024 = assign_custom_weeks(real_2024)
        real_2024w = real_2024.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index()
        real_2024w.rename(columns={'tgmv':'yoy'}, inplace=True)
    else:
        real_2024w = pd.DataFrame(columns=['week','week_start','week_end','yoy'])

    # --- Prophet para predecir ---
    df_prophet = real_2025[['date','tgmv']].dropna()
    df_prophet = df_prophet.rename(columns={'date':'ds','tgmv':'y'})
    
    if len(df_prophet) > 2:  # evita fallo si hay pocos datos
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=n_pred_weeks*7)
        forecast = model.predict(future)
    
        forecast = forecast[['ds','yhat']].rename(columns={'ds':'date'})
        forecast['date'] = pd.to_datetime(forecast['date'])
        forecast = assign_custom_weeks(forecast)
        forecast_weekly = forecast.groupby(['week','week_start','week_end'])['yhat'].sum().reset_index()
    else:
        forecast_weekly = pd.DataFrame(columns=['week','week_start','week_end','yhat'])


    # --- Merge todo ---
    df_weeks = plan_weekly.merge(real_2025w, on=['week','week_start','week_end'], how='left')
    df_weeks = df_weeks.merge(real_2024w[['week','yoy']], on='week', how='left')
    df_weeks = df_weeks.merge(forecast_weekly[['week','yhat']], on='week', how='left')

    # Métricas
    df_weeks['diff_yoy'] = df_weeks['real'] - df_weeks['yoy']
    df_weeks['cumplimiento'] = df_weeks['real'] / df_weeks['plan'] * 100

    # Futuro/pasado
    df_weeks['is_future'] = df_weeks['real'].isna()
    df_weeks['diff_display'] = np.where(df_weeks['is_future'],
                                        df_weeks['yhat']-df_weeks['yoy'],
                                        df_weeks['diff_yoy'])
    df_weeks['cumplimiento_display'] = np.where(df_weeks['is_future'],
                                                df_weeks['yhat']/df_weeks['plan']*100,
                                                df_weeks['cumplimiento'])

    # Colores para diff
    def color_diff(row):
        if pd.isna(row['diff_display']):
            return "gray"
        if row['is_future']:
            return "lightgreen" if row['diff_display'] > 0 else "lightcoral"
        else:
            return "green" if row['diff_display'] > 0 else "red"
    df_weeks['color_diff'] = df_weeks.apply(color_diff, axis=1)

    # Colores para cumplimiento
    def color_cump(row):
        if pd.isna(row['cumplimiento_display']):
            return "gray"
        if row['is_future']:
            return "lightgreen" if row['cumplimiento_display'] >= 100 else "lightsalmon"
        else:
            return "green" if row['cumplimiento_display'] >= 100 else "orange"
    df_weeks['color_cump'] = df_weeks.apply(color_cump, axis=1)

    # Etiquetas
    df_weeks['semana_lbl'] = df_weeks.apply(
        lambda x: f"Semana {x['week']} ({x['week_start'].date()} a {x['week_end'].date()})", axis=1
    )

    # --- Vista previa ---
    st.subheader(":mag: Vista previa de las semanas agregadas")
    st.dataframe(df_weeks[['week','week_start','week_end','plan','real','yoy','yhat','diff_display','cumplimiento_display']].head(20))

    # --- Gráfico 1: Plan vs Real vs Predicción ---
    st.subheader("📈 Evolución semanal Plan vs Real vs Predicción (interactivo)")
    melted = df_weeks.melt(
        id_vars=['week','semana_lbl','is_future'],
        value_vars=['real','plan','yhat'],
        var_name='Métrica', value_name='Valor'
    )
    chart1 = (
        alt.Chart(melted)
        .mark_line(point=True)
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("Valor:Q", title="TGMV"),
            color=alt.Color('Métrica:N', legend=alt.Legend(title="Métrica")),
            tooltip=['semana_lbl','Métrica','Valor'],
            strokeDash=alt.condition("datum.Métrica == 'yhat'", alt.value([5,5]), alt.value([0]))
        )
        .properties(height=400, width=850)
    )
    st.altair_chart(chart1, use_container_width=True)

    # --- Gráfico 2: Diferencia Real/Pred vs YoY ---
    st.subheader("📉 Diferencia Real/Pred vs YoY")
    chart2 = (
        alt.Chart(df_weeks)
        .mark_bar()
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("diff_display:Q", title="Diferencia"),
            color=alt.Color('color_diff:N', legend=None),
            tooltip=['semana_lbl','real','yoy','yhat','diff_display']
        )
        .properties(height=350, width=850)
    )
    st.altair_chart(chart2, use_container_width=True)

    # --- Gráfico 3: Cumplimiento vs Plan ---
    st.subheader("📊 Cumplimiento vs Plan (%)")
    chart3 = (
        alt.Chart(df_weeks)
        .mark_bar()
        .encode(
            x=alt.X("week:O", title="Semana"),
            y=alt.Y("cumplimiento_display:Q", title="% Cumplimiento"),
            color=alt.Color('color_cump:N', legend=None),
            tooltip=['semana_lbl','real','plan','yhat','cumplimiento_display']
        )
        .properties(height=350, width=850)
    )
    st.altair_chart(chart3, use_container_width=True)

    # Botón descarga
    st.download_button(
        "Descargar datos semanales",
        df_weeks.to_csv(index=False).encode('utf-8'),
        file_name="datos_dashboard_semanal.csv",
        mime="text/csv"
    )
else:
    st.warning("Por favor sube ambos archivos CSV para continuar.")
