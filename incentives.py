import streamlit as st
import pandas as pd
import altair as alt

# --- Estilo CSS personalizado ---
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #FF6600;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        font-size: 16px;
        padding: 10px 24px;
        transition: background-color 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #e65c00;
    }
    .main-header {
        color: #FF6600;
        font-weight: 900;
        font-size: 50px;
        text-align: center;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-header {
        color: #FF6600;
        font-weight: 700;
        font-size: 28px;
        margin-top: 30px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .section-title {
        text-align: center;
        font-size: 22px;
        color: #555;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Título principal
st.markdown("<h1 class='main-header'>Calculadora DXGY 🧮🍊</h1>", unsafe_allow_html=True)

# 1. Upload CSV ciudades
st.markdown("<h4 class='section-title'>📊 1. Carga el CSV de ciudades</h4>", unsafe_allow_html=True)
df_ciudades = st.file_uploader("Sube el CSV de ciudades (city_id, city_name, gmv, country_code, cluster)", type="csv")

if df_ciudades is not None:
    df_c = pd.read_csv(df_ciudades, encoding='latin-1')
    df_c.columns = df_c.columns.str.strip().str.lower()

    # 2. Mostrar SQL dinámico
    st.markdown("<h4 class='section-title'>📄 2. SQL Query para generar el CSV de viajes</h4>", unsafe_allow_html=True)
    clusters = sorted(df_c['cluster'].dropna().unique())
    cluster_seleccionado = st.selectbox("Selecciona tu cluster", clusters)

    ciudades_cluster = df_c[df_c['cluster'] == cluster_seleccionado]
    city_ids = ciudades_cluster['city_id'].tolist()
    country_codes = ciudades_cluster['country_code'].unique().tolist()

    city_ids_str = ', '.join(map(str, city_ids))
    country_codes_str = ", ".join(f"'{c}'" for c in country_codes)

    with st.expander("📋 Ver SQL Query"):
        st.code(f"""
SELECT
    pt,
    city_id,
    driver_id,
    FROM_UNIXTIME(
        UNIX_TIMESTAMP(finish_time) - INT(SUBSTR(stat_start_hour, -2, 2)) * 3600,
        'HH'
    ) AS trip_hour,
    COUNT(DISTINCT order_id) AS trips
FROM international_capital.dwm_trd_order_pro_core_anycar_base_di
WHERE pt = 20250623
    AND country_code IN ({country_codes_str})
    AND city_id IN ({city_ids_str})
    AND is_td_finish = 1
GROUP BY
    pt, city_id, driver_id,
    FROM_UNIXTIME(
        UNIX_TIMESTAMP(finish_time) - INT(SUBSTR(stat_start_hour, -2, 2)) * 3600,
        'HH'
    );
        """, language="sql")

    # 3. Upload CSV viajes
    st.markdown("<h4 class='section-title'>📈 3. Carga el CSV de viajes</h4>", unsafe_allow_html=True)
    df_viajes = st.file_uploader("Sube el CSV de viajes (pt, city_id, driver_id, trip_hour, trips, final_cohort, shor, asp, ipt, gmv)", type="csv")

    if df_viajes is not None:
        df_vi = pd.read_csv(df_viajes)
        df_vi.columns = df_vi.columns.str.strip().str.lower()
        

        # --- Multiple select cohort ---
        cohorts_disponibles = sorted(df_vi['final_cohort'].fillna('No Cohort').unique())
        cohort_seleccionados = st.sidebar.multiselect("Selecciona cohorts a incentivar", cohorts_disponibles, default=cohorts_disponibles)

        # Filtrar por cohorts seleccionados
        df_v = df_vi[df_vi['final_cohort'].isin(cohort_seleccionados)]

        # Sidebar inputs
        st.sidebar.header("🏙️ Selección de ciudad y horario")
        ciudades_disponibles = ciudades_cluster[['city_name', 'city_id', 'gmv', 'country_code']]
        directorio_ciudades = {
            row['city_name']: {
                "city_id": row['city_id'],
                "gmv": row['gmv'],
                "country_code": row['country_code']
            } for _, row in ciudades_disponibles.iterrows()
        }
        ciudad_seleccionada = st.sidebar.selectbox("Selecciona la ciudad", list(directorio_ciudades.keys()))
        city_info = directorio_ciudades[ciudad_seleccionada]
        city_id = city_info['city_id']
        country_code = city_info['country_code']

        hora_inicio = st.sidebar.number_input("Hora inicio (0-23)", min_value=0, max_value=23, value=9)
        hora_fin = st.sidebar.number_input("Hora fin (0-23)", min_value=0, max_value=23, value=12)

        st.sidebar.header("⚙️ Tipo de Incentivo")
        tipo_incentivo = st.sidebar.selectbox("Selecciona el tipo de incentivo", ["DXGY", "Multiplier"])

        st.sidebar.header("🎯 Define los Tiers")
        num_tiers = st.sidebar.slider("Número de Tiers", min_value=1, max_value=6, value=3)

        tiers_manual = []
        for i in range(num_tiers):
            viajes = st.sidebar.number_input(f"Viajes Tier {i+1}", min_value=1, value=5 + i*2, key=f"viajes_t{i}")

            if tipo_incentivo == "DXGY":
                reward = st.sidebar.number_input(f"Reward por viaje Tier {i+1} ($)", min_value=0, value=600 + i*200, key=f"reward_t{i}")
                tiers_manual.append({"viajes": viajes, "reward": reward})

            elif tipo_incentivo == "Multiplier":
                multiplier = st.sidebar.number_input(f"Multiplier Tier {i+1} (%)", min_value=0.0, max_value=100.0, value=20.0 + i*10, step=1.0, key=f"multiplier_t{i}")
                tiers_manual.append({"viajes": viajes, "multiplier": multiplier / 100})

        burn_objetivo = st.sidebar.number_input("🎯 Burn objetivo % (opcional)", min_value=0.0, max_value=100.0, value=5.0)

        # Filtramos solo datos de la ciudad seleccionada y cohorts seleccionados
        df_dia = df_v[df_v['city_id'] == city_id]

        # Agrupamos por hora para total viajes
        trips_por_hora = df_dia.groupby('trip_hour')['trips'].sum().reset_index()

        # Agrupamos por hora para contar conductores únicos activos en esa hora
        conductores_por_hora = df_dia.groupby('trip_hour')['driver_id'].nunique().reset_index()
        conductores_por_hora.rename(columns={'driver_id': 'conductores_activos'}, inplace=True)

        # Unimos ambos DataFrames para calcular TPH por hora
        tph_por_hora_df = pd.merge(trips_por_hora, conductores_por_hora, on='trip_hour', how='left')

        # Calculamos TPH = trips / conductores activos en esa hora
        tph_por_hora_df['conductores_activos'] = tph_por_hora_df['conductores_activos'].replace(0, pd.NA)
        tph_por_hora_df['tph'] = tph_por_hora_df['trips'] / tph_por_hora_df['conductores_activos']

        # Mostrar tabla
        st.markdown("---")
        st.markdown(f"<h4 style='color:#FF6600; text-align:center;'>Información y Visuales</h4>", unsafe_allow_html=True)
        st.markdown(f"<h4 class='section-title'>⚙️ {ciudad_seleccionada} Tabla de Información</h4>", unsafe_allow_html=True)
        st.dataframe(
            tph_por_hora_df.rename(columns={
                'trip_hour': 'Hora',
                'trips': 'Total de viajes',
                'conductores_activos': 'Conductores activos',
                'tph': 'TPH'
            }),
            use_container_width=True,
            height=250
        )

        # Gráficos Trips por hora
        st.markdown(f"<h4 class='section-title'>🚖 {ciudad_seleccionada} Total de Viajes Por Hora</h4>", unsafe_allow_html=True)
        chart_viajes = alt.Chart(tph_por_hora_df).mark_bar(color='#f97316').encode(
            x=alt.X('trip_hour', title='Hora del día'),
            y=alt.Y('trips', title='Total de viajes'),
            tooltip=['trip_hour', 'trips']
        ).properties(width=650, height=300)
        st.altair_chart(chart_viajes, use_container_width=True)

        # Gráfico Drivers
        st.markdown(f"<h4 class='section-title'>🚙 {ciudad_seleccionada} Total de DRVS Por Hora</h4>", unsafe_allow_html=True)
        chart_drivers = alt.Chart(tph_por_hora_df).mark_bar(color='#3b82f6').encode(
            x=alt.X('trip_hour', title='Hora del día'),
            y=alt.Y('conductores_activos', title='Total de drivers'),
            tooltip=['trip_hour', 'conductores_activos']
        ).properties(width=650, height=300)
        st.altair_chart(chart_drivers, use_container_width=True)

        # Gráfico TPH
        st.markdown(f"<h4 class='section-title'>🚃 {ciudad_seleccionada} TPH Por Hora</h4>", unsafe_allow_html=True)
        chart_tph = alt.Chart(tph_por_hora_df).mark_bar(color='#74c476').encode(
            x=alt.X('trip_hour', title='Hora del día'),
            y=alt.Y('tph', title='TPH'),
            tooltip=['trip_hour', 'tph']
        ).properties(width=650, height=300)
        st.altair_chart(chart_tph, use_container_width=True)

        # Gráfico distribución de viajes por conductor en horario seleccionado
        df_filtrado_horas = df_dia[(df_dia['trip_hour'] >= hora_inicio) & (df_dia['trip_hour'] < hora_fin)]
        df_conductor = df_filtrado_horas.groupby('driver_id')['trips'].sum().reset_index()

        distribucion = df_conductor['trips'].value_counts().sort_index().reset_index()
        distribucion.columns = ['Viajes', 'Conductores']

        st.markdown(f"<h4 class='section-title'>📊 Distribución de viajes por conductor entre {hora_inicio}:00 y {hora_fin}:00</h4>", unsafe_allow_html=True)
        chart = alt.Chart(distribucion).mark_bar(color="#f97316").encode(
            x=alt.X('Viajes:O', title='Número de viajes'),
            y=alt.Y('Conductores:Q', title='Número de conductores'),
            tooltip=['Viajes', 'Conductores']
        ).properties(width=600, height=300)
        st.altair_chart(chart, use_container_width=True)

        # --- Cálculo del TPH incentivo y mínimo de viajes estimados ---
        df_incentivo = df_filtrado_horas
        df_por_hora_incentivo = df_incentivo.groupby('trip_hour').agg(
            total_trips=('trips', 'sum'),
            drivers_hora=('driver_id', 'nunique')
        ).reset_index()

        suma_trips_incentivo = df_por_hora_incentivo['total_trips'].sum()
        suma_drivers_incentivo = df_por_hora_incentivo['drivers_hora'].sum()

        tph_incentivo = suma_trips_incentivo / suma_drivers_incentivo if suma_drivers_incentivo > 0 else 0
        horas_incentivo = hora_fin - hora_inicio
        minimo_trips_estimado = round(tph_incentivo * horas_incentivo)

        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚀 TPH en horas del incentivo", f"{tph_incentivo:.2f}")
            st.metric("📌 Mínimo de trips estimado", f"{minimo_trips_estimado} viajes")
        promedio_asp = df_filtrado_horas['asp'].mean() if not df_filtrado_horas.empty else 0
        promedio_ipt = df_filtrado_horas['ipt'].mean() if not df_filtrado_horas.empty else 0

        
        col_asp, col_ipt = st.columns(2)
        with col_asp:
            st.metric("💵 ASP promedio ($)", f"{promedio_asp:.2f}")
        with col_ipt:
            st.metric("🛠️ IPT promedio ($)", f"{promedio_ipt:.2f}")

        # --- Lógica cálculo incentivos ---
        df_conductor = df_filtrado_horas.groupby('driver_id')['trips'].sum().reset_index()

        tiers = sorted(tiers_manual, key=lambda x: x['viajes'])
        conductores_por_tier = [0] * len(tiers)

        for _, row in df_conductor.iterrows():
            viajes = row['trips']
            for i in range(len(tiers)):
                if viajes >= tiers[i]['viajes']:
                    conductores_por_tier[i] += 1

        conductores_exclusivos = []
        for i in range(len(tiers)):
            if i == len(tiers) - 1:
                exclusivos = conductores_por_tier[i]
            else:
                exclusivos = conductores_por_tier[i] - conductores_por_tier[i + 1]
            conductores_exclusivos.append(exclusivos)

        reward_acumulado_por_tier = []
        avg_ipt = df_filtrado_horas['ipt'].mean() if not df_filtrado_horas.empty else 0

        if tipo_incentivo == "DXGY":
            for i in range(len(tiers)):
                reward_total = 0
                for j in range(i + 1):
                    viajes_incrementales = tiers[j]['viajes'] if j == 0 else tiers[j]['viajes'] - tiers[j - 1]['viajes']
                    reward_total += viajes_incrementales * tiers[j]['reward']
                reward_acumulado_por_tier.append(reward_total)

        elif tipo_incentivo == "Multiplier":
            for i in range(len(tiers)):
                viajes_incrementales = tiers[i]['viajes'] if i == 0 else tiers[i]['viajes'] - tiers[i - 1]['viajes']
                multiplier = tiers[i]['multiplier']
                cap = round(avg_ipt * multiplier)
                reward_total = viajes_incrementales
                reward_total = viajes_incrementales * cap
                if i > 0:
                    reward_total += reward_acumulado_por_tier[i - 1]
                reward_acumulado_por_tier.append(reward_total)

        gmv_total_real = df_vi[df_vi['city_id'] == city_id]['gmv'].sum()
        total_burn = sum(reward_acumulado_por_tier[i] * conductores_exclusivos[i] for i in range(len(tiers)))
        porcentaje_burn_real = (total_burn / gmv_total_real) * 100 if gmv_total_real > 0 else 0
        conductores_calificados = sum(conductores_exclusivos)
        win_rate = conductores_calificados / df_conductor.shape[0] if df_conductor.shape[0] > 0 else 0

        st.markdown("---")
        st.markdown(f"<h4 class='section-title'>📈 Resultados del Incentivo</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🚗 Conductores calificados", conductores_calificados)
            st.metric("🔥 % Burn Real", f"{porcentaje_burn_real:.2f}%")
        with col2:
            st.metric("💸 Burn total ($)", f"{total_burn:,.2f}")
            st.metric("🎯 % Burn Target", f"{burn_objetivo:.2f}%")
        with col3:
            st.metric("🏁 Win Rate", f"{win_rate:.2%}")
            delta_burn = burn_objetivo - porcentaje_burn_real
            color = "inverse" if delta_burn < 0 else "normal"
            st.metric("📉 Burn vs Target", value=f"{delta_burn:+.2f}%", delta=f"{delta_burn:+.2f}%", delta_color=color)
        with col4:
            st.metric("💰 GMV Real", f"{gmv_total_real:,.2f}")

        # --- Desglose por Tier ---
        st.markdown("---")
        st.markdown("<h3 class='sub-header'>📊 Desglose por Tier</h3>", unsafe_allow_html=True)

        porcentaje_sobre_calificados = [
            (c / conductores_calificados if conductores_calificados > 0 else 0)
            for c in conductores_exclusivos
        ]

        df_resultado = pd.DataFrame({
            "Tier": [f"Tier {i+1}" for i in range(len(tiers))],
            "Viajes requeridos": [t['viajes'] for t in tiers],
            "Conductores exclusivos": conductores_exclusivos,
            "Reward acumulado": reward_acumulado_por_tier,
            "Burn por tier": [reward_acumulado_por_tier[i] * conductores_exclusivos[i] for i in range(len(tiers))],
            "% sobre calificados": [f"{p:.1%}" for p in porcentaje_sobre_calificados]
        })

        if tipo_incentivo == "DXGY":
            df_resultado["Reward por viaje"] = [t['reward'] for t in tiers]
        elif tipo_incentivo == "Multiplier":
            df_resultado["Multiplier %"] = [f"{t['multiplier']*100:.0f}%" for t in tiers]
            df_resultado["Cap por viaje"] = [round(avg_ipt * t['multiplier']) for t in tiers]

        st.dataframe(df_resultado, use_container_width=True, height=300)

        # --- Formato incremental para copiar/pegar ---
        st.markdown("---")
        st.markdown("### 📝 Formato incremental para copiar y pegar:")

        if country_code == "MX":
            sufijo = "MXN$"
        elif country_code == "CR":
            sufijo = "₡"
        else:
            sufijo = "$"

        formato_tiers_incremental = []

        for i, t in enumerate(tiers):
            viajes_incrementales = t['viajes'] if i == 0 else t['viajes'] - tiers[i - 1]['viajes']
            if tipo_incentivo == "DXGY":
                reward_incremental = viajes_incrementales * t['reward']
                formato_tiers_incremental.append(f"Tier{i+1}: {viajes_incrementales}Trips*{reward_incremental:.0f}{sufijo}")
            elif tipo_incentivo == "Multiplier":
                multiplier_pct = int(t['multiplier'] * 100)
                cap = round(avg_ipt * t['multiplier'])
                formato_tiers_incremental.append(f"Tier{i+1}: {viajes_incrementales}Trips*{multiplier_pct}%ASP*Cap{cap}{sufijo}")

        formato_tiers_str = ", ".join(formato_tiers_incremental)
        st.code(formato_tiers_str, language="")

        # --- Footer ---
        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#888888; font-size:16px;'>"
            "© 2025 DXGY/Multiplier Calculator - Hecho con ❤️ por POC"
            "</p>", unsafe_allow_html=True
        )
