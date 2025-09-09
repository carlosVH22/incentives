import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet
import requests
import io

# --- Página ---
st.set_page_config(page_title="Dashboard Incentivos", layout="wide") 

# --- CSS para diseño moderno ---
st.markdown("""
<style>
    /* Fondo de la app */
    .stApp {
        background-color: #f7f8fa;
    }

    /* Contenedores estilo card */
    .card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    /* Títulos grandes y coloridos */
    .section-title {
        color: #4B0082;
        font-size: 22px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    /* Subtítulos de explicación */
    .section-subtitle {
        color: #555;
        font-size: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# --- Header central ---
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h1 style='color:#4B0082;'>📊 Dashboard de Incentivos TGMV 2025</h1>
    <p style='color:#555;'>Comparación Real vs Plan vs Predicción con YoY</p>
</div>
""", unsafe_allow_html=True)
# URLs de los CSVs
PLAN_URL = "https://raw.githubusercontent.com/carlosVH22/incentives/refs/heads/main/df_plan%20(1).csv"
REAL_URL = "https://raw.githubusercontent.com/carlosVH22/incentives/refs/heads/main/df_tgmv_9.csv"

# Guardar los CSV "crudos" en las variables
plan_file = io.StringIO(requests.get(PLAN_URL).text)
real_file = io.StringIO(requests.get(REAL_URL).text)


# --- Función semanas custom ---
def assign_custom_weeks(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    year = df['date'].dt.year.iloc[0]
    all_days = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    start = all_days[0]
    while start.weekday() != 5 and start <= all_days[-1]:  # sábado
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

# --- Función Prophet con ajuste y futuro ---
def weekly_forecast_with_adjustment_and_future(df, start_sunday='2025-01-05', final_date='2025-12-31'):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    last_actual_date = df['Date'].max()
    final_date = pd.Timestamp(final_date)
    sundays = pd.date_range(start=pd.Timestamp(start_sunday), end=final_date, freq='W-SUN')

    weekly_rows = []
    daily_preds_list = []

    model_kwargs = dict(seasonality_mode='additive',
                        yearly_seasonality=True,
                        changepoint_prior_scale=0.001,
                        seasonality_prior_scale=4,
                        n_changepoints=50)

    for s in sundays:
        train = df[df['Date'] < s].rename(columns={'Date':'ds','TGMV':'y'})[['ds','y']].dropna()
        if len(train)<2:
            continue
        week_end = s + pd.Timedelta(days=6)
        model = Prophet(**model_kwargs)
        model.fit(train)
        future = pd.DataFrame({'ds': pd.date_range(start=s, end=week_end)})
        forecast = model.predict(future)[['ds','yhat']].copy()
        baseline_week_pred = forecast['yhat'].sum()

        st_end_for_factor = s + pd.Timedelta(days=2)
        actual_st = df[(df['Date']>=s)&(df['Date']<=st_end_for_factor)][['Date','TGMV']].rename(columns={'Date':'ds','TGMV':'actual'})
        cmp_df = forecast.merge(actual_st,on='ds',how='left')
        cmp_df['pct_diff'] = np.where((~cmp_df['actual'].isna())&(cmp_df['yhat']!=0),
                                      (cmp_df['actual']/cmp_df['yhat'])-1, np.nan)
        cmp_df = cmp_df[cmp_df['ds']<=st_end_for_factor]
        n_days_factor = int(cmp_df['pct_diff'].notna().sum())
        factor = cmp_df['pct_diff'].mean() if n_days_factor>0 else np.nan
        adjusted_week_pred = baseline_week_pred*(1+factor) if pd.notna(factor) else baseline_week_pred
        week_actual = df[(df['Date']>=s)&(df['Date']<=week_end)]['TGMV']
        actual_week_total = week_actual.sum() if len(week_actual)==7 else np.nan

        dtmp = forecast.rename(columns={'yhat':'pred'})
        dtmp['week_start'] = s
        daily_preds_list.append(dtmp)

        weekly_rows.append({'week_start':s,
                            'week_end':week_end,
                            'baseline_week_pred':baseline_week_pred,
                            'adjustment_factor_SunMonTue':factor,
                            'n_days_used_for_factor':n_days_factor,
                            'adjusted_week_pred_hist':adjusted_week_pred,
                            'actual_week_total':actual_week_total})
    weekly_df = pd.DataFrame(weekly_rows).sort_values('week_start').reset_index(drop=True)
    daily_df = pd.concat(daily_preds_list,ignore_index=True) if daily_preds_list else pd.DataFrame()

    # Promedios históricos
    valid_factors = weekly_df['adjustment_factor_SunMonTue'].dropna()
    avg_all = valid_factors.mean() if len(valid_factors)>0 else 0
    neg_factors = valid_factors[valid_factors<0]
    pos_factors = valid_factors[valid_factors>0]
    avg_neg = neg_factors.mean() if len(neg_factors)>0 else avg_all
    avg_pos = pos_factors.mean() if len(pos_factors)>0 else avg_all
    weekly_df['is_future'] = weekly_df['week_start']>last_actual_date
    weekly_df['proj_general'] = np.where(weekly_df['adjustment_factor_SunMonTue'].notna(),
                                         weekly_df['baseline_week_pred']*(1+weekly_df['adjustment_factor_SunMonTue']),
                                         np.where(weekly_df['is_future'],
                                                  weekly_df['baseline_week_pred']*(1+avg_all),
                                                  weekly_df['baseline_week_pred']))
    weekly_df['proj_neg'] = np.where(weekly_df['is_future'],weekly_df['baseline_week_pred']*(1+avg_neg),np.nan)
    weekly_df['proj_pos'] = np.where(weekly_df['is_future'],weekly_df['baseline_week_pred']*(1+avg_pos),np.nan)

    return weekly_df, daily_df

# --- Main ---
if plan_file and real_file:
    df_plan = pd.read_csv(plan_file)
    df_plan.columns = df_plan.columns.str.strip().str.lower()
    df_plan['date'] = pd.to_datetime(df_plan['date'])
    df_plan = assign_custom_weeks(df_plan)
    plan_weekly = df_plan.groupby(['week','week_start','week_end'])['plan'].sum().reset_index()

    df_real = pd.read_csv(real_file)
    df_real.columns = df_real.columns.str.strip().str.lower()
    df_real['date'] = pd.to_datetime(df_real['date'])
    df_real['Date'] = df_real['date']  # para Prophet
    df_real['TGMV'] = df_real['tgmv']  # para Prophet

    # Forecast con Prophet
    weekly_prophet, daily_prophet = weekly_forecast_with_adjustment_and_future(df_real)

    # --- Agregamos columnas para comparación ---
    real_2025 = df_real[df_real['date'].dt.year==2025].copy()
    real_2025 = assign_custom_weeks(real_2025)
    real_2025w = real_2025.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index().rename(columns={'tgmv':'real'})
    df_weeks = plan_weekly.merge(real_2025w,on=['week','week_start','week_end'],how='left')
    df_weeks = df_weeks.merge(weekly_prophet[['week_start','proj_general','proj_neg','proj_pos','is_future']],on='week_start',how='left')
    df_weeks['diff_pred'] = df_weeks['proj_general'] - df_weeks['plan']
    df_weeks['cumplimiento'] = df_weeks['real']/df_weeks['plan']*100
    df_weeks['cumplimiento_future'] = df_weeks['proj_general']/df_weeks['plan']*100
    df_weeks['semana_lbl'] = df_weeks.apply(lambda x: f"Semana {x['week']} ({x['week_start'].date()} a {x['week_end'].date()})", axis=1)


    # --- Gráfico Real vs Plan vs Predicción con rango optimista/pesimista ---
    
    selection = alt.selection_point(fields=['week'])
    
    # Filtrar semanas futuras
    df_future = df_weeks[df_weeks['is_future'].fillna(False)]
    
    # Real en columnas
    real_chart = alt.Chart(df_weeks).mark_bar(opacity=0.5).encode(
        x='week:O',
        y=alt.Y('real:Q', title='TGMV', axis=alt.Axis(format=',')),
        tooltip=['week','semana_lbl','real:Q'],
        color=alt.value('#1f77b4'),
        opacity=alt.condition(selection, alt.value(0.7), alt.value(0.5))
    )
    
    # Plan en barras grises
    plan_chart = alt.Chart(df_weeks).mark_bar(opacity=0.8).encode(
        x='week:O',
        y=alt.Y('plan:Q', axis=alt.Axis(format=',')),
        tooltip=['week','semana_lbl','plan:Q'],
        color=alt.value('lightgray'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.5))
    )
    
    # Área sombreada optimista/pesimista (solo semanas futuras)
    band_chart = alt.Chart(df_future).mark_area(opacity=0.5, color="lightblue").encode(
        x='week:O',
        y=alt.Y('proj_neg:Q', axis=alt.Axis(format=',')),
        y2='proj_pos:Q',
        tooltip=['week','semana_lbl','proj_neg:Q','proj_pos:Q']
    )
    
    # Predicción: línea punteada + puntos rellenos
    pred_chart = alt.layer(
        alt.Chart(df_weeks).mark_line(strokeDash=[5,5], color='red').encode(
            x='week:O',
            y=alt.Y('proj_general:Q', axis=alt.Axis(format=',')),
            tooltip=['week','semana_lbl','proj_general:Q']
        ),
        alt.Chart(df_weeks).mark_point(filled=True, size=15, stroke='red', color='red').encode(
            x='week:O',
            y='proj_general:Q',
            tooltip=['week','semana_lbl','proj_general:Q']
        )
    )
    
    # Combinar todos los charts
    chart = alt.layer(
        plan_chart,
        real_chart,
        band_chart,
        pred_chart
    ).add_params(selection).interactive().properties(height=400, width=850)
    



    
    # --- Gráfico Cumplimiento vs Plan ---
    
    # Crear columna de visualización según si es futuro o pasado
    df_weeks['cumplimiento_display'] = np.where(
        df_weeks['is_future'],
        df_weeks['cumplimiento_future'],
        df_weeks['cumplimiento']
    )
    
    # Columna con color según reglas
    def color_rule(row):
        if row['is_future']:
            return 'lightblue'  # futuro
        elif row['cumplimiento_display'] >= 100:
            return 'green'
        else:
            return 'orange'
    
    df_weeks['color_cumplimiento'] = df_weeks.apply(color_rule, axis=1)
    
    # Selección interactiva
    selection2 = alt.selection_point(fields=['week'])
    
    # Chart Altair
    chart2 = (
        alt.Chart(df_weeks)
        .mark_bar()
        .encode(
            x=alt.X('week:O', title='Semana'),
            y=alt.Y('cumplimiento_display:Q', title='% Cumplimiento Plan'),
            color=alt.Color('color_cumplimiento:N', legend=None),
            tooltip=[
                alt.Tooltip('week:O', title='Semana'),
                alt.Tooltip('semana_lbl:N', title='Rango de fechas'),
                alt.Tooltip('plan:Q', title='Plan', format=','),
                alt.Tooltip('real:Q', title='Real', format=','),
                alt.Tooltip('proj_general:Q', title='Proyección', format=','),
                alt.Tooltip('cumplimiento_display:Q', title='% Cumplimiento', format='.1f')
            ],
            opacity=alt.condition(selection2, alt.value(1), alt.value(0.7))
        )
        .add_params(selection2)
        .interactive()
        .properties(height=350, width=850)
    )
    


    real_2024 = df_real[df_real['date'].dt.year==2024].copy()
    real_2024 = assign_custom_weeks(real_2024)
    real_2024w = real_2024.groupby(['week','week_start','week_end'])['tgmv'].sum().reset_index().rename(columns={'tgmv':'real_2024'})

    # Unir semanas 2024 y 2025
    df_yoy = df_weeks.merge(real_2024w[['week','real_2024']], on='week', how='left')

    # Calcular cumplimiento YoY
    df_yoy['cumplimiento_yoy'] = np.where(
        df_yoy['is_future'],
        (df_yoy['proj_general'] / df_yoy['real_2024']-1) * 100,  # Futuro -> proyección
        (df_yoy['real'] / df_yoy['real_2024']-1) * 100           # Pasado -> real
    )


    def color_rule_yoy(row):
        if row['is_future']:
            return 'lightblue'   # futuro
        elif row['cumplimiento_yoy'] >= 0:
            return 'green'
        else:
            return 'orange'
    
    df_yoy['color_yoy'] = df_yoy.apply(color_rule_yoy, axis=1)

    
    selection3 = alt.selection_point(fields=['week'])
    
    chart3 = (
        alt.Chart(df_yoy)
        .mark_bar()
        .encode(
            x=alt.X('week:O', title='Semana', axis=alt.Axis(format=',')),
            y=alt.Y('cumplimiento_yoy:Q', title='% Cumplimiento YoY (2025 vs 2024)', axis=alt.Axis(format=',')),
            color=alt.Color('color_yoy:N', legend=None),
            tooltip=[
                alt.Tooltip('week:O', title='Semana'),
                alt.Tooltip('semana_lbl:N', title='Rango de fechas'),
                alt.Tooltip('real_2024:Q', title='Real 2024', format=','),
                alt.Tooltip('real:Q', title='Real 2025', format=','),
                alt.Tooltip('proj_general:Q', title='Proyección 2025', format=','),
                alt.Tooltip('cumplimiento_yoy:Q', title='% Cumplimiento YoY', format='.1f')
            ],
            opacity=alt.condition(selection3, alt.value(1), alt.value(0.7))
        )
        .add_params(selection3)
        .interactive()
        .properties(height=350, width=850)
    )



    # --- Contenedores estilo card para secciones ---
    
    # --- Vista Semanal (tabla) ---
    with st.container():
        st.markdown("""
        <div class='card'>
            <div class='section-title'>📊 Vista Semanal</div>
            <div class='section-subtitle'>Tabla resumen de Plan, Real y Predicción semanal</div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(
            df_weeks.style.format({
                "plan": "{:,.1f}",
                "real": "{:,.1f}",
                "proj_general": "{:,.1f}",
                "proj_neg": "{:,.1f}",
                "proj_pos": "{:,.1f}",
                "diff_pred": "{:,.1f}",
                "cumplimiento": "{:,.1f}",
                "cumplimiento_future": "{:,.1f}"
            })
        )
    
    # --- Gráfico Real vs Plan vs Predicción ---
    with st.container():
        st.markdown("""
        <div class='card'>
            <div class='section-title'>📊 Gráfico Real vs Plan vs Predicción</div>
            <div class='section-subtitle'>Incluye rango optimista/pesimista para semanas futuras</div>
        </div>
        """, unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)
    
    # --- Gráfico Cumplimiento vs Plan (%) ---
    with st.container():
        st.markdown("""
        <div class='card'>
            <div class='section-title'>📊 Cumplimiento vs Plan (%)</div>
            <div class='section-subtitle'>Se muestran valores reales y proyecciones para semanas futuras</div>
        </div>
        """, unsafe_allow_html=True)
        st.altair_chart(chart2, use_container_width=True)
    
    # --- Gráfico YoY 2025 vs 2024 ---
    with st.container():
        st.markdown("""
        <div class='card'>
            <div class='section-title'>📊 YoY 2025 vs 2024</div>
            <div class='section-subtitle'>Comparación de TGMV real y proyectado respecto a 2024</div>
        </div>
        """, unsafe_allow_html=True)
        st.altair_chart(chart3, use_container_width=True)



    import datetime

    # Fecha de hoy
    today = pd.Timestamp.today()
    
    # Detectar semana en curso
    current_week = df_weeks[(df_weeks['week_start'] <= today) & (df_weeks['week_end'] >= today)]
    if not current_week.empty:
        week_start = current_week['week_start'].iloc[0]
        week_end = current_week['week_end'].iloc[0]
        week_number = current_week['week'].iloc[0]
    
        # Datos diarios de la semana actual
        week_days = df_real[(df_real['date'] >= week_start) & (df_real['date'] <= today)]
    
        parcial_real = week_days['TGMV'].sum()
        dias_completados = len(week_days)
        total_dias_semana = (week_end - week_start).days + 1
    
        # Estimación para la semana completa
        tendencia_diaria = parcial_real / dias_completados if dias_completados > 0 else 0
        estimado_semana_completa = tendencia_diaria * total_dias_semana
    
        # Plan de la semana
        plan_semana = current_week['plan'].iloc[0]
        cumplimiento_est = estimado_semana_completa / plan_semana * 100
    
        # YoY estimado
        real_2024_semana = df_yoy[df_yoy['week']==week_number]['real_2024'].iloc[0]
        yoy_est = (estimado_semana_completa / real_2024_semana - 1) * 100

    # --- Cards con columnas para estimaciones de la semana en curso ---
    if not current_week.empty:
        col1, col2 = st.columns(2)
    
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Estimación Cumplimiento al Plan</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Semana en curso basada en tendencia diaria</div>', unsafe_allow_html=True)
            
            est_chart = alt.Chart(pd.DataFrame({
                'Semana':[f"Semana {week_number}"],
                'Cumplimiento Estimado':[cumplimiento_est]
            })).mark_bar(color='#8A2BE2', opacity=0.8).encode(
                x='Semana:N',
                y=alt.Y('Cumplimiento Estimado:Q', title='% Cumplimiento Plan'),
                tooltip=[alt.Tooltip('Cumplimiento Estimado:Q', format='.1f')]
            ).properties(height=300, width=400)
            
            st.altair_chart(est_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Estimación Crecimiento YoY</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-subtitle">Semana en curso basada en tendencia diaria</div>', unsafe_allow_html=True)
            
            yoy_chart = alt.Chart(pd.DataFrame({
                'Semana':[f"Semana {week_number}"],
                'YoY Estimado':[yoy_est]
            })).mark_bar(color='#20B2AA', opacity=0.8).encode(
                x='Semana:N',
                y=alt.Y('YoY Estimado:Q', title='% Crecimiento YoY'),
                tooltip=[alt.Tooltip('YoY Estimado:Q', format='.1f')]
            ).properties(height=300, width=400)
            
            st.altair_chart(yoy_chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)





