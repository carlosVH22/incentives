import streamlit as st
import pandas as pd
from datetime import datetime, time

# --- Estilo CSS personalizado para botones y texto ---
st.markdown(
    """
    <style>
    /* Botones con color naranja y bordes redondeados */
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
    /* Centramos y estilizamos el header principal */
    .main-header {
        color: #FF6600;
        font-weight: 900;
        font-size: 45px;
        text-align: center;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Subtítulos */
    .sub-header {
        color: #FF6600;
        font-weight: 700;
        font-size: 28px;
        margin-top: 30px;
        margin-bottom: 10px;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Pequeño espacio entre métricas */
    .metric-container {
        margin-bottom: 10px;
    }
    /* Código con fondo sutil y sombra */
    .code-block {
        background-color: #f9f9f9;
        border-radius: 8px;
        padding: 10px 16px;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
        font-size: 16px;
        font-family: Consolas, monospace;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar para inputs
import streamlit as st
import pandas as pd
from datetime import datetime, time

# --- Estilo CSS personalizado ---
st.markdown("""
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
.metric-container {
    margin-bottom: 10px;
}
.code-block {
    background-color: #f9f9f9;
    border-radius: 8px;
    padding: 10px 16px;
    box-shadow: 0 0 8px rgba(0,0,0,0.05);
    font-size: 16px;
    font-family: Consolas, monospace;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# --- Título ---
st.markdown("<h1 class='main-header'>Calculadora DXGY 🧾🍊</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <h3 style='text-align: center; color: #333333; font-size: 24px; font-weight: 600;'>
        📥 Carga de archivos
    </h3>
    """,
    unsafe_allow_html=True
)

# --- Input CSV de ciudades (necesario antes de generar query) ---
df_ciudades = st.file_uploader("🌐 Carga el CSV de ciudades (city_id, city_name, gmv, country_code, cluster)", type="csv")

# --- Selección de cluster y generación dinámica de query ---
cluster_opciones = ['Heros', 'Growers', 'Rocket', 'POC LAB', 'POC Academy', 'Superman']
cluster_seleccionado = st.selectbox("🧱 Selecciona el cluster para generar la SQL", cluster_opciones)

city_ids_str = "-- carga primero el CSV de ciudades --"
countries_str = "-- carga primero el CSV de ciudades --"

if df_ciudades:
    df_c = pd.read_csv(df_ciudades, encoding='latin-1')
    df_c.columns = df_c.columns.str.strip().str.lower()
    df_cluster = df_c[df_c['cluster'] == cluster_seleccionado]
    city_ids_cluster = df_cluster['city_id'].dropna().astype(int).tolist()
    country_codes_cluster = df_cluster['country_code'].dropna().unique().tolist()

    city_ids_str = ", ".join(str(cid) for cid in city_ids_cluster)
    countries_str = "', '".join(country_codes_cluster)

with st.expander("📄 SQL Query para generar el CSV de viajes (según el cluster seleccionado)"):
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
    AND country_code IN ('{countries_str}')
    AND city_id IN (
        {city_ids_str}
    )
    AND is_td_finish = 1
GROUP BY
    pt,
    city_id,
    driver_id,
    FROM_UNIXTIME(
        UNIX_TIMESTAMP(finish_time) - INT(SUBSTR(stat_start_hour, -2, 2)) * 3600,
        'HH'
    );
""", language="sql")

# --- Input CSV de viajes ---
df_viajes = st.file_uploader("🚊 Carga el CSV de viajes (pt, city_id, driver_id, trip_hour, trips)", type="csv")
if df_viajes:
    df_v = pd.read_csv(df_viajes)
    # Diccionario de ciudades
    directorio_ciudades = {
        row['city_name']: {"city_id": row['city_id'], "gmv": row['gmv'], "country_code": row['country_code']}
        for _, row in df_c.iterrows()
    }
    
    st.sidebar.markdown(
    """
    <h3 style='text-align: center; color: #333333; font-size: 22px; font-weight: 600;'>
        🧪 Información del incentivo
    </h3>
    """,
    unsafe_allow_html=True
    )
    st.sidebar.header("🏙️ Selección de ciudad y horario")
    ciudad_seleccionada = st.sidebar.selectbox("Selecciona la ciudad", list(directorio_ciudades.keys()))
    city_info = directorio_ciudades[ciudad_seleccionada]
    city_id = city_info["city_id"]
    gmv = city_info["gmv"]
    country_code = city_info["country_code"]

    hora_inicio = st.sidebar.number_input("Hora inicio", min_value=0, max_value=23, value=9)
    hora_fin = st.sidebar.number_input("Hora fin", min_value=0, max_value=23, value=12)

    st.sidebar.header("🎯 Define los Tiers")
    num_tiers = st.sidebar.slider("Niveles (TIRs)", 1, 6, 3)

    tiers_manual = []
    for i in range(int(num_tiers)):
        viajes = st.sidebar.number_input(f"Viajes Tier {i+1}", min_value=1, value=5 + i*2, key=f"viajes_t{i}")
        reward = st.sidebar.number_input(f"Reward por viaje Tier {i+1} ($)", min_value=0, value=600 + i*200, key=f"reward_t{i}")
        tiers_manual.append({"viajes": viajes, "reward": reward})

    burn_objetivo = st.sidebar.number_input("🎯 Burn objetivo % (opc)", min_value=0.0, max_value=100.0, value=5.0)

    if st.sidebar.button("✅ Calcular Burn"):
        # --- PROCESAMIENTO ---
        df_dia = df_v[df_v['city_id'] == city_id]
        total_conductores_dia = df_dia['driver_id'].nunique()

        df_filtrado = df_dia[(df_dia['trip_hour'] >= hora_inicio) & (df_dia['trip_hour'] < hora_fin)]
        df_conductor = df_filtrado.groupby('driver_id')['trips'].sum().reset_index()

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
        for i in range(len(tiers)):
            reward_total = 0
            for j in range(i + 1):
                if j == 0:
                    viajes = tiers[j]['viajes']
                else:
                    viajes = tiers[j]['viajes'] - tiers[j - 1]['viajes']
                reward_total += viajes * tiers[j]['reward']
            reward_acumulado_por_tier.append(reward_total)

        total_burn = sum(reward_acumulado_por_tier[i] * conductores_exclusivos[i] for i in range(len(tiers)))
        porcentaje_burn_real = (total_burn / gmv) * 100 if gmv > 0 else 0
        conductores_calificados = sum(conductores_exclusivos)
        win_rate = conductores_calificados / total_conductores_dia if total_conductores_dia > 0 else 0

        # --- RESULTADOS ---
        st.markdown("---")
        st.success("✅ Resultado del cálculo:")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚗 Conductores calificados", conductores_calificados)
            st.metric("🔥 % Burn Real", f"{porcentaje_burn_real:.2f}%")
        with col2:
            st.metric("💸 Burn total ($)", f"{total_burn:,.2f}")
            st.metric("🎯 % Burn Target", f"{burn_objetivo:.2f}%")
        with col3:
            st.metric("🏁 Win Rate", f"{win_rate:.2%}")

            delta_burn = burn_objetivo - porcentaje_burn_real
            color = "normal"  # verde si delta positivo, rojo si negativo

            st.metric(
                "📉 Burn vs Traget",
                value=f"{delta_burn:+.2f}%",
                delta=f"{delta_burn:+.2f}%",
                delta_color=color
            )
            
        st.markdown("---")
        st.markdown("<h3 class='sub-header'>📊 Desglose por Tier</h3>", unsafe_allow_html=True)

        porcentaje_sobre_calificados = [
            (c / conductores_calificados if conductores_calificados > 0 else 0)
            for c in conductores_exclusivos
        ]

        df_resultado = pd.DataFrame({
            "Tier": [f"Tier {i+1}" for i in range(len(tiers))],
            "Viajes requeridos": [t['viajes'] for t in tiers],
            "Reward por viaje": [t['reward'] for t in tiers],
            "Conductores exclusivos": conductores_exclusivos,
            "Reward acumulado": reward_acumulado_por_tier,
            "Burn por tier": [reward_acumulado_por_tier[i] * conductores_exclusivos[i] for i in range(len(tiers))],
            "% sobre calificados": [f"{p:.1%}" for p in porcentaje_sobre_calificados]
        })

        st.dataframe(df_resultado, use_container_width=True, height=280)

        st.markdown("---")
        st.markdown("### 📝 Formato incremental para copiar y pegar:")

        # Definir sufijo según country_code
        if country_code == "MX":
            sufijo = "*30MXN$"
        elif country_code == "CR":
            sufijo = "*3000₡"
        else:
            sufijo = "$"  # default para CL, CO, u otros

        formato_tiers_incremental = []
        for i, t in enumerate(tiers):
            if i == 0:
                viajes_incrementales = t['viajes']
            else:
                viajes_incrementales = t['viajes'] - tiers[i-1]['viajes']
            reward_incremental = viajes_incrementales * t['reward']
            formato_tiers_incremental.append(f"Tier{i+1}:{t['viajes']}Trips*{reward_incremental:.0f}{sufijo}")

        formato_tiers_str = ", ".join(formato_tiers_incremental)
        st.code(formato_tiers_str, language="")

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center; color:#888888; font-size:16px;'>"
            "© 2025 DXGY Calculator - Hecho con ❤️ por POC"
            "</p>", unsafe_allow_html=True
        )
