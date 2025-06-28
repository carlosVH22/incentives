#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Calculadora de Incentivos DiDi", layout="centered")

st.markdown("""
    <style>
        body {
            background-color: #fff8f0;
        }
        .main {
            background-color: #fff8f0;
            color: #333;
        }
        .stButton>button {
            background-color: #f97316;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-weight: bold;
        }
        .block-container {
            padding-top: 2rem;
        }
        h1, .stMarkdown h1 {
            font-family: 'Helvetica Neue', sans-serif;
        }
        .stDataFrame th, .stDataFrame td {
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='color:#f97316;text-align:center;'>🧮 Calculadora de Incentivos DiDi</h1>
    <p style='text-align:center;color:#666;'>Optimiza tus programas de incentivos y genera el resumen para copiar y pegar 🚗</p>
""", unsafe_allow_html=True)

# Parte 1: Parámetros de cálculo de incentivos
st.sidebar.header("Parámetros Generales")
city = st.sidebar.text_input("Ciudad", value="Temuco")
GMV = st.sidebar.number_input("GMV total", value=1000000.0, step=10000.0)
burn_rate = st.sidebar.number_input("Burn rate (%)", value=5.0, step=0.5) / 100
cohort_size = st.sidebar.number_input("Tamaño del cohort", value=500, step=10)
win_rate = st.sidebar.number_input("Win rate (%)", value=18.0, step=1.0) / 100
TPH = st.sidebar.number_input("Trips por hora (TPH)", value=2.0)
horas_incentivo = st.sidebar.number_input("Horas incentivadas", value=3.0)
IPT = st.sidebar.number_input("Ingreso por viaje (IPT)", value=166.67)

st.sidebar.header("Tipo de Incentivo")
tipo_incentivo = st.sidebar.selectbox("Selecciona tipo de incentivo", ['dxgy', 'multiplier', 'guaranteed'])

st.sidebar.header("Tiers de viajes")
num_tiers = st.sidebar.number_input("Cantidad de TIRs", min_value=1, max_value=10, value=3, step=1)
TIRs_info = []
for i in range(num_tiers):
    viajes = st.sidebar.number_input(f"Viajes requeridos para TIR #{i+1}", min_value=1, value=8 + i * 2)
    TIRs_info.append(viajes)

# Cálculo de incentivos
@st.cache_data
def calcular_incentivo(GMV, burn_rate, cohort_size, win_rate,
                       TPH, horas_incentivo, IPT, tipo_incentivo, TIRs_info):
    budget_total = GMV * burn_rate
    ganadores = int(cohort_size * win_rate)
    presupuesto_por_conductor = budget_total / ganadores
    viajes_base = TPH * horas_incentivo
    valor_base = viajes_base * IPT

    resultados = []
    esfuerzos = []
    for viajes_requeridos in TIRs_info:
        valor_estimado = viajes_requeridos * IPT
        esfuerzo = max(0, valor_estimado - valor_base)
        esfuerzos.append(esfuerzo)

    total_esfuerzo = sum(esfuerzos)
    pesos = [1 / len(TIRs_info)] * len(TIRs_info) if total_esfuerzo == 0 else [e / total_esfuerzo for e in esfuerzos]

    for i, viajes_requeridos in enumerate(TIRs_info):
        valor_estimado = viajes_requeridos * IPT
        esfuerzo_incremental = esfuerzos[i]
        peso = pesos[i]

        if tipo_incentivo == 'dxgy':
            incentivo_estimado = presupuesto_por_conductor * peso
        elif tipo_incentivo == 'multiplier':
            porcentaje = peso * 100
            incentivo_estimado = valor_estimado * (porcentaje / 100)
        elif tipo_incentivo == 'guaranteed':
            ingreso_garantizado = valor_estimado + (presupuesto_por_conductor * peso)
            incentivo_estimado = ingreso_garantizado - valor_estimado
        else:
            raise ValueError("Tipo de incentivo no reconocido")

        resultados.append({
            "Viajes": viajes_requeridos,
            "Valor estimado ($)": round(valor_estimado, 2),
            "Esfuerzo incremental ($)": round(esfuerzo_incremental, 2),
            "Peso esfuerzo": round(peso, 4),
            "Incentivo estimado ($)": round(incentivo_estimado, 2),
            "Presupuesto sugerido por conductor ($)": round(presupuesto_por_conductor, 2)
        })

    return resultados

# Función para construir la regla del evento
@st.cache_data
def construir_regla_evento(tipo, tiers, incentivos, IPT, TPH):
    reglas = []
    for i, viajes in enumerate(tiers):
        recompensa = round(incentivos[i], 2)
        if tipo == 'dxgy':
            reglas.append(f"Tier{i+1}:{viajes}Trips*{int(recompensa)}$")
        elif tipo == 'multiplier':
            cap = round(recompensa)
            porcentaje = int((recompensa / (viajes * IPT)) * 100)
            reglas.append(f"Tier{i+1}: {viajes}Trips*{porcentaje}%ASP*Cap{cap}$")
        elif tipo == 'guaranteed':
            extra = round(recompensa - viajes * IPT, 2)
            garantizado = round(viajes * IPT + extra, 2)
            reglas.append(f"Tier{i+1}: {viajes}Trips*{int(viajes/TPH)}Hour*Guarantee{int(garantizado)}$*Extra{int(extra)}$")
    return ",".join(reglas)

if st.button("Calcular Incentivos"):
    resultados = calcular_incentivo(
        GMV, burn_rate, cohort_size, win_rate,
        TPH, horas_incentivo, IPT, tipo_incentivo, TIRs_info
    )

    df_resultados = pd.DataFrame(resultados)
    st.subheader("📊 Resultados por TIR")
    st.dataframe(df_resultados, use_container_width=True)

    incentivos_list = df_resultados["Incentivo estimado ($)"].tolist()
    regla_evento = construir_regla_evento(tipo_incentivo, TIRs_info, incentivos_list, IPT, TPH)

    st.subheader("📋 Event Rules (copia y pega)")
    st.code(regla_evento, language='markdown')

