import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet

st.set_page_config(page_title="Dashboard Incentivos", layout="wide")
st.title("📊 Dashboard de TGMV: Real vs Plan vs Predicción")

# --- Sidebar para CSVs ---
st.sidebar.header("Carga los archivos CSV")
plan_file = st.sidebar.file_uploader("Sube el plan diario 2025 (date, plan)", type="csv")
real_file = st.sidebar.file_uploader("Sube los datos reales (date, tgmv), 2024 y 2025", type="csv")

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

    st.subheader("📊 Vista semanal (plan, real y predicción)")
    st.dataframe(df_weeks.head(10))


    # --- Gráfico Real vs Plan vs Predicción ---
    selection = alt.selection_point(fields=['week'])
    
    # Real en columnas
    real_chart = alt.Chart(df_weeks).mark_bar(opacity=0.8).encode(
        x=alt.X('week:O', title='Semana'),
        y=alt.Y('real:Q', title='TGMV'),
        tooltip=['week','semana_lbl','real:Q'],
        color=alt.value('#1f77b4'),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.7))
    )
    
    # Plan en área gris
    plan_chart = alt.Chart(df_weeks).mark_area(opacity=0.5, color='lightgray').encode(
        x='week:O',
        y='plan:Q',
        tooltip=['week','semana_lbl','plan:Q']
    )
    
    # Predicción en línea punteada
    pred_chart = alt.Chart(df_weeks).mark_line(point=True, strokeDash=[5,5], color='red').encode(
        x='week:O',
        y='proj_general:Q',
        tooltip=['week','semana_lbl','proj_general:Q']
    )
    
    # Combinar todos los gráficos
    chart = (real_chart + plan_chart + pred_chart).add_params(selection).interactive().properties(height=400,width=850)
    
    st.altair_chart(chart, use_container_width=True)
    
    # --- Gráfico Cumplimiento vs Plan ---
    st.subheader("📊 Cumplimiento vs Plan (%) (real y futuro)")
    
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
    
    st.altair_chart(chart2, use_container_width=True)
