import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# =========================
# Función principal (adaptada de tu versión)
# =========================
def weekly_forecast_with_adjustment_and_future(
    df,
    start_sunday='2025-01-05',
    end_sunday=None,
    final_date='2025-12-31',
    model_kwargs=None
):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if end_sunday is None:
        end_sunday = df['Date'].max()
    start_sunday = pd.Timestamp(start_sunday)
    end_sunday = pd.Timestamp(end_sunday)
    final_date = pd.Timestamp(final_date)
    horizon_end = max(end_sunday, final_date)
    sundays = pd.date_range(start=start_sunday, end=horizon_end, freq='W-SUN')

    if model_kwargs is None:
        model_kwargs = dict(
            seasonality_mode='additive',
            yearly_seasonality=True,
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=4,
            n_changepoints=50
        )

    weekly_rows = []
    daily_preds_list = []
    last_actual_date = df['Date'].max()

    for s in sundays:
        train = (
            df[df['Date'] < s]
            .rename(columns={'Date': 'ds', 'TGMV': 'y'})[['ds', 'y']]
            .sort_values('ds')
            .dropna()
        )
        if len(train) < 2:
            continue

        week_end = s + pd.Timedelta(days=6)

        model = Prophet(**model_kwargs)
        model.fit(train)

        future = pd.DataFrame({'ds': pd.date_range(start=s, end=week_end, freq='D')})
        forecast = model.predict(future)[['ds', 'yhat']].copy()

        baseline_week_pred = forecast['yhat'].sum()

        # Ajuste Sun-Mon-Tue
        st_end_for_factor = s + pd.Timedelta(days=2)
        actual_st = df[(df['Date'] >= s) & (df['Date'] <= st_end_for_factor)][['Date', 'TGMV']].copy()
        actual_st = actual_st.rename(columns={'Date': 'ds', 'TGMV': 'actual'})
        cmp_df = forecast.merge(actual_st, on='ds', how='left')
        cmp_df['pct_diff'] = np.where(
            (~cmp_df['actual'].isna()) & (cmp_df['yhat'] != 0),
            (cmp_df['actual'] / cmp_df['yhat']) - 1.0,
            np.nan
        )
        cmp_df = cmp_df[cmp_df['ds'] <= st_end_for_factor]
        n_days_factor = int(cmp_df['pct_diff'].notna().sum())
        factor = cmp_df['pct_diff'].mean() if n_days_factor > 0 else np.nan

        adjusted_week_pred = baseline_week_pred * (1.0 + factor) if pd.notna(factor) else baseline_week_pred

        week_actual = df[(df['Date'] >= s) & (df['Date'] <= week_end)]['TGMV']
        actual_week_total = week_actual.sum() if len(week_actual) == 7 else np.nan

        weekly_rows.append({
            'week_start': s,
            'week_end': week_end,
            'baseline_week_pred': baseline_week_pred,
            'adjustment_factor_SunMonTue': factor,
            'n_days_used_for_factor': n_days_factor,
            'adjusted_week_pred_hist': adjusted_week_pred,
            'actual_week_total': actual_week_total
        })

        dtmp = forecast[['ds', 'yhat']].copy()
        dtmp['week_start'] = s
        dtmp.rename(columns={'yhat': 'pred'}, inplace=True)
        daily_preds_list.append(dtmp)

    weekly_df = pd.DataFrame(weekly_rows).sort_values('week_start').reset_index(drop=True)
    daily_df = pd.concat(daily_preds_list, ignore_index=True) if daily_preds_list else pd.DataFrame()
    if weekly_df.empty:
        return weekly_df, daily_df

    # Calcular promedios
    valid_factors = weekly_df['adjustment_factor_SunMonTue'].dropna()
    avg_all = valid_factors.mean() if len(valid_factors) > 0 else 0.0
    avg_neg = valid_factors[valid_factors < 0].mean() if (valid_factors < 0).any() else avg_all
    avg_pos = valid_factors[valid_factors > 0].mean() if (valid_factors > 0).any() else avg_all

    weekly_df['is_future'] = weekly_df['week_start'] > last_actual_date
    weekly_df['proj_general'] = np.where(
        weekly_df['adjustment_factor_SunMonTue'].notna(),
        weekly_df['baseline_week_pred'] * (1.0 + weekly_df['adjustment_factor_SunMonTue']),
        np.where(weekly_df['is_future'],
                 weekly_df['baseline_week_pred'] * (1.0 + avg_all),
                 weekly_df['baseline_week_pred'])
    )
    weekly_df['proj_neg'] = np.where(weekly_df['is_future'], weekly_df['baseline_week_pred'] * (1.0 + avg_neg), np.nan)
    weekly_df['proj_pos'] = np.where(weekly_df['is_future'], weekly_df['baseline_week_pred'] * (1.0 + avg_pos), np.nan)

    weekly_df = weekly_df[weekly_df['week_start'] <= final_date].reset_index(drop=True)
    return weekly_df, daily_df


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Forecast Dashboard", layout="wide")
st.title("📊 Weekly Forecast with Adjustment & YoY Comparison")

# Inputs
uploaded_hist = st.file_uploader("Carga datos históricos (Date, TGMV)", type=["csv"])
uploaded_yoy = st.file_uploader("Carga datos YoY y Plan (week_start, YoY, Plan)", type=["csv"])
start_sunday = st.date_input("Fecha inicial (domingo)", pd.to_datetime("2025-01-05"))
final_date = st.date_input("Fecha final de proyección", pd.to_datetime("2025-12-31"))

if uploaded_hist:
    df = pd.read_csv(uploaded_hist)
    weekly_df, daily_df = weekly_forecast_with_adjustment_and_future(
        df=df,
        start_sunday=start_sunday,
        final_date=final_date
    )

    st.subheader("Resultados semanales")
    st.dataframe(weekly_df)

    # Gráfico 1: Predicción + Intervalo
    st.subheader("Predicción con ajuste Sun-Mon-Tue y promedios futuros")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(weekly_df['week_start'], weekly_df['proj_general'], label="Proyección (promedio general)", color="blue")
    if weekly_df['proj_neg'].notna().any():
        ax.fill_between(
            weekly_df['week_start'], weekly_df['proj_neg'], weekly_df['proj_pos'],
            alpha=0.2, label="Rango negativo/positivo"
        )
    if weekly_df['actual_week_total'].notna().any():
        ax.scatter(weekly_df['week_start'], weekly_df['actual_week_total'], label="Real", color="black", s=20)
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

    # Si hay datos YoY y Plan
    if uploaded_yoy:
        yoy_df = pd.read_csv(uploaded_yoy)
        yoy_df['week_start'] = pd.to_datetime(yoy_df['week_start'])

        merged = weekly_df.merge(yoy_df, on="week_start", how="left")

        st.subheader("Comparación YoY / Plan")
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        ax2.plot(merged['week_start'], merged['YoY'], label="YoY (año pasado)", linestyle="--", color="orange")
        ax2.plot(merged['week_start'], merged['proj_general'], label="Predicción", color="blue")
        ax2.plot(merged['week_start'], merged['Plan'], label="Plan", color="green")
        if merged['actual_week_total'].notna().any():
            ax2.scatter(merged['week_start'], merged['actual_week_total'], label="Real", color="black", s=20)
        ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig2)

        # KPIs
        st.subheader("📌 Indicadores clave")
        merged['diff_vs_plan'] = merged['proj_general'] - merged['Plan']
        merged['diff_vs_yoy'] = merged['proj_general'] - merged['YoY']
        col1, col2 = st.columns(2)
        col1.metric("Promedio diferencia vs Plan", f"{merged['diff_vs_plan'].mean():,.0f}")
        col2.metric("Promedio diferencia vs YoY", f"{merged['diff_vs_yoy'].mean():,.0f}")

        # Descargar resultados
        st.download_button(
            "📥 Descargar resultados en CSV",
            merged.to_csv(index=False).encode("utf-8"),
            file_name="weekly_forecast_results.csv",
            mime="text/csv"
        )
