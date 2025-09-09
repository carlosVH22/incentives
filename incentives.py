# ==========================
# Prophet integrado
# ==========================
from prophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def weekly_forecast_with_adjustment_and_future(
    df,
    start_sunday='2025-01-05',
    end_sunday=None,
    final_date='2025-12-31',
    plot=True,
    model_kwargs=None
):
    """
    df: DataFrame con columnas ['Date', 'TGMV'] en frecuencia diaria.
    Ajusta semanalmente usando el factor Sun-Mon-Tue cuando existen reales.
    """
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

        dtmp = forecast[['ds', 'yhat']].copy()
        dtmp['week_start'] = s
        dtmp.rename(columns={'yhat': 'pred'}, inplace=True)
        daily_preds_list.append(dtmp)

        weekly_rows.append({
            'week_start': s,
            'week_end': week_end,
            'baseline_week_pred': baseline_week_pred,
            'adjustment_factor_SunMonTue': factor,
            'n_days_used_for_factor': n_days_factor,
            'adjusted_week_pred_hist': adjusted_week_pred,
            'actual_week_total': actual_week_total
        })

    weekly_df = pd.DataFrame(weekly_rows).sort_values('week_start').reset_index(drop=True)
    daily_df = pd.concat(daily_preds_list, ignore_index=True) if daily_preds_list else pd.DataFrame()

    if weekly_df.empty:
        return weekly_df, daily_df

    valid_factors = weekly_df['adjustment_factor_SunMonTue'].dropna()
    if len(valid_factors) > 0:
        avg_all = valid_factors.mean()
        avg_neg = valid_factors[valid_factors < 0].mean() if (valid_factors < 0).any() else avg_all
        avg_pos = valid_factors[valid_factors > 0].mean() if (valid_factors > 0).any() else avg_all
    else:
        avg_all = avg_neg = avg_pos = 0.0

    weekly_df['is_future'] = weekly_df['week_start'] > last_actual_date

    weekly_df['proj_general'] = np.where(
        weekly_df['adjustment_factor_SunMonTue'].notna(),
        weekly_df['baseline_week_pred'] * (1.0 + weekly_df['adjustment_factor_SunMonTue']),
        np.where(
            weekly_df['is_future'],
            weekly_df['baseline_week_pred'] * (1.0 + avg_all),
            weekly_df['baseline_week_pred']
        )
    )

    weekly_df['proj_neg'] = np.where(
        weekly_df['is_future'],
        weekly_df['baseline_week_pred'] * (1.0 + avg_neg),
        np.nan
    )
    weekly_df['proj_pos'] = np.where(
        weekly_df['is_future'],
        weekly_df['baseline_week_pred'] * (1.0 + avg_pos),
        np.nan
    )

    if horizon_end > final_date:
        weekly_df = weekly_df[weekly_df['week_start'] <= final_date].reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(weekly_df['week_start'], weekly_df['proj_general'], label='Proyección (promedio general)')
        future_mask = weekly_df['is_future']
        if future_mask.any():
            ax.fill_between(
                weekly_df.loc[future_mask, 'week_start'],
                weekly_df.loc[future_mask, 'proj_neg'],
                weekly_df.loc[future_mask, 'proj_pos'],
                alpha=0.25,
                label='Rango (neg ↔ pos)'
            )
        if weekly_df['actual_week_total'].notna().any():
            ax.scatter(
                weekly_df.loc[weekly_df['actual_week_total'].notna(), 'week_start'],
                weekly_df.loc[weekly_df['actual_week_total'].notna(), 'actual_week_total'],
                label='Real',
                s=20
            )
        ax.set_title('Proyección semanal con ajuste Sun–Mon–Tue')
        ax.set_xlabel('Semana (inicio)')
        ax.set_ylabel('TGMV semanal')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

    return weekly_df, daily_df
