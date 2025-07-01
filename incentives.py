import streamlit as st
import pandas as pd
from datetime import datetime, time

# Estilo personalizado
st.set_page_config(page_title="Calculadora de Incentivos DiDi", layout="centered")
st.markdown("""
    <style>
        body { background-color: #fff8f0; }
        .main { background-color: #fff8f0; color: #333; }
        .stButton>button {
            background-color: #f97316;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .stNumberInput>div>input { border-radius: 8px; }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Reinicio manual
if st.sidebar.button("🔁 Reiniciar incentivos acumulados"):
    st.session_state.batch_acumulado = []
    st.sidebar.success("Acumulador reiniciado correctamente.")

if 'batch_acumulado' not in st.session_state:
    st.session_state.batch_acumulado = []

# ================= LÓGICA =================

def calcular_incentivo(GMV, burn_rate, TPH, horas_incentivo, IPT, TIRs_info, winners_por_tier):
    budget_total = GMV * burn_rate
    baseline_viajes = TPH * horas_incentivo

    esfuerzos_tiers = []
    for viajes, winners in zip(TIRs_info, winners_por_tier):
        esfuerzo_individual = max(0, viajes - baseline_viajes)
        esfuerzo_total = esfuerzo_individual * winners
        esfuerzos_tiers.append(esfuerzo_total)

    total_esfuerzo = sum(esfuerzos_tiers)
    valor_por_unidad_esfuerzo = budget_total / total_esfuerzo if total_esfuerzo > 0 else 0

    incentivos = []
    for esfuerzo_total, winners in zip(esfuerzos_tiers, winners_por_tier):
        incentivo_total_tier = esfuerzo_total * valor_por_unidad_esfuerzo
        incentivo_por_conductor = incentivo_total_tier / winners if winners > 0 else 0
        incentivos.append(incentivo_por_conductor)

    return incentivos, None  # El segundo valor se mantiene por compatibilidad

def construir_regla_evento(tipo, tiers, incentivos, IPT, TPH):
    reglas = []
    for i, viajes in enumerate(tiers):
        recompensa = round(incentivos[i], 2)
        if tipo == 'dxgy':
            reglas.append(f"Tier{i+1}:{viajes}Trips*{int(recompensa)}$")
        elif tipo == 'multiplier':
            porcentaje = int((recompensa / (viajes * IPT)) * 100) if viajes * IPT > 0 else 0
            cap = round(porcentaje / 100 * IPT)
            reglas.append(f"Tier{i+1}: {viajes}Trips*{porcentaje}%ASP*Cap{cap}$")
        elif tipo == 'guaranteed':
            extra = round(recompensa, 2)
            garantizado = viajes * IPT + extra
            reglas.append(f"Tier{i+1}: {viajes}Trips*{int(viajes/TPH)}Hour*Guarantee{int(garantizado)}$*Extra{int(extra)}$")
    return ",".join(reglas)

def limpiar_emojis(texto):
    return texto.encode('latin-1', 'ignore').decode('latin-1')

# ============ INTERFAZ PRINCIPAL =============

st.markdown("""
    <h1 style='color:#f97316;text-align:center;'>🧮 Calculadora de Incentivos DiDi</h1>
    <p style='text-align:center;color:#666;'>Optimiza tus programas de incentivos por TIRs 🚗</p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📊 Parámetros Generales")
city = st.sidebar.text_input("Ciudad", value="Ciudad X")
GMV = st.sidebar.number_input("GMV total", value=1000000.0, min_value=0.0)
burn_rate = st.sidebar.number_input("Burn rate (%)", value=5.0, min_value=0.0, max_value=100.0) / 100
cohort_size = st.sidebar.number_input("Cohort size", value=500, min_value=1)
win_rate = st.sidebar.number_input("Win rate (%)", value=18.0, min_value=0.0, max_value=100.0) / 100
TPH = st.sidebar.number_input("TPH", value=2.0, min_value=0.0)
horas = st.sidebar.number_input("Horas de incentivo", value=3.0, min_value=0.0)
IPT = st.sidebar.number_input("Ingreso por viaje", value=160.0, min_value=0.0)
AR = st.sidebar.number_input("AR", value=80.0, min_value=0.0)
CR = st.sidebar.number_input("CR", value=80.0, min_value=0.0)
target_group = st.sidebar.number_input("Target Group", value=80.0, min_value=0.0)

st.sidebar.header("🎯 Configura el Incentivo")
tipo_incentivo = st.sidebar.selectbox("Tipo", ["dxgy", "multiplier", "guaranteed"])
num_tiers = st.sidebar.slider("Niveles (TIRs)", 1, 6, 3)
tiers = [st.sidebar.number_input(f"Viajes TIR #{i+1}", min_value=1, value=8 + i*2) for i in range(num_tiers)]
winners_por_tier = [st.sidebar.number_input(f"Winners TIR #{i+1}", min_value=1, value=10) for i in range(num_tiers)]


# Inputs para fechas y horas fuera del botón, en la página principal (no sidebar)
st.markdown("### ⏳ Configuración de Periodos")
reward_day = st.text_input("📅 Reward Period (YYYY-MM-DD)", value=datetime.today().strftime('%Y-%m-%d'))

col1, col2 = st.columns(2)
with col1:
    start_time = st.time_input("Hora inicio evento", value=time(4, 0), step=60)
with col2:
    end_time = st.time_input("Hora fin evento", value=time(23, 59), step=60)

event_period = f"{start_time.strftime('%H:%M')}-{end_time.strftime('%H:%M')}"
push_time = f"{reward_day} {start_time.strftime('%H:%M')}"

# Botón principal para calcular y agregar
if st.button("✅ Calcular incentivo y agregar al CSV"):
    incentivos, por_conductor = calcular_incentivo(GMV, burn_rate, TPH, horas, IPT, tiers, winners_por_tier)
    regla = construir_regla_evento(tipo_incentivo, tiers, incentivos, IPT, TPH)

    title_map = {
        "dxgy": "¡Obten hasta  {{maximum_total_amount}} adicionales!",
        "multiplier": "¡Multiplica tus ganacias hasta x!",
        "guaranteed": "¡Ganacias garantizadas por!"
    }

    push_map = {
        "dxgy": "Conéctate a DiDi y obtén hasta {{maximum_total_amount}} adicionales.",
        "multiplier": "Conéctate a DiDi y multiplica tus ganancias hasta x.",
        "guaranteed": "Conéctate a DiDi y obtén ingresos garantizados por x"
    }

    st.session_state.batch_acumulado.append({
        "Event Purpose": "Active DRV",
        "City": city,
        "Driver Type": "Express",
        "Event Type": {
            "dxgy": "RealtimeDxGyReward",
            "multiplier": "Multiplier",
            "guaranteed": "GuaranteeReward"
        }[tipo_incentivo],
        "Trigger Type": "Trip",
        "Budget Department": "Engage-Cityops",
        "Budget": round(GMV * burn_rate/100, 2),
        "Select Period": "",
        "Reward Period": reward_day,
        "Event Period": event_period,
        "Target Group": target_group,
        "Ratio of Control Group": 10,
        "Trip Type": "Express,DiDi Entrega,Set Your Fare",
        "AR & CR Calculation Type": "Daily" if tipo_incentivo == "dxgy" else "Normal Logic",
        "AR": AR,
        "CR": CR,
        "Star Rating": 3.5,
        "Geofence": "",
        "Event Rules": regla,
        "Title": title_map[tipo_incentivo],
        "Notes": "{{reward_rules}",
        "Comments": "",
        "Push1_Send Time": push_time,
        "Push1_Send push to specified driver(s)": "Drivers participating in the event",
        "Push1_Content": push_map[tipo_incentivo],
        "Push2_Send Time": "",
        "Push2_Send push to specified driver(s)": "",
        "Push2_Content": ""
    })

    st.success("Incentivo agregado exitosamente ✅")
# =========== VISTA DE INCENTIVOS ACUMULADOS ==========

if st.session_state.batch_acumulado:
    st.markdown("---")
    st.subheader("📦 Incentivos acumulados en esta sesión")

    df_acumulado = pd.DataFrame(st.session_state.batch_acumulado)
    
    # Seleccionamos solo columnas relevantes para visualizar
    columnas_resumen = [
        "City", "Reward Period", "Event Period", "Driver Type",
        "Event Type", "Budget", "Event Rules", "Push1_Send Time"
    ]
    df_mostrar = df_acumulado[columnas_resumen]
    
    st.dataframe(df_mostrar, use_container_width=True)

# =========== DESCARGA Y RESETEO ==========

if st.button("📥 Descargar CSV con incentivos acumulados"):
    df = pd.DataFrame(st.session_state.batch_acumulado)
    for col in ["Title", "Push1_Content"]:
        df[col] = df[col].apply(limpiar_emojis)

    st.download_button(
        "Descargar archivo CSV",
        data=df.to_csv(index=False).encode("latin-1"),
        file_name="incentivos_batch.csv",
        mime="text/csv"
    )

    if st.checkbox("¿Reiniciar acumulador después de descargar?"):
        st.session_state.batch_acumulado = []
        st.success("Acumulador reiniciado luego de la descarga ✔️")
