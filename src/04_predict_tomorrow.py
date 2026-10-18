"""
03_predict_tomorrow.py
Prototipo experimental de inferencia diaria (t+1) basado solo en pronóstico meteorológico GFS.
El desempeño en vivo aún no está medido: cada corrida se guarda en un historial para evaluarlo.
"""
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import yaml

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN Y RUTAS
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("InferenciaExperimental")

# Detección flexible de raíz (funciona ejecutado desde la raíz o desde src/)
CURRENT_DIR = Path().resolve()
BASE_DIR = CURRENT_DIR if (CURRENT_DIR / "stations.yaml").exists() else CURRENT_DIR.parent

CONFIG_PATH = BASE_DIR / "stations.yaml"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
HISTORIAL_CSV = PREDICTIONS_DIR / "historial_predicciones.csv"

# Debe coincidir con el TAG con que se guardaron los modelos (celda de serialización).
TAG_MODELO = "v2"

# Ciudades donde el modelo no se distingue del azar en el test (ver README).
CIUDADES_NO_CONFIABLES = {
    "Santiago": "AUC ~0.62 en test; no distinguible del azar solo con pronóstico",
}

# Variables que se calculan aquí; el modelo usa las que declare su artefacto.
FEATURES_ESPERADAS = [
    "temp_min_day_forecast",
    "wind_mean_day_forecast",
    "temperature_2m_noche_forecast",
    "wind_speed_10m_noche_forecast",
    "delta_temp_noche",
    "delta_wind_noche",
    "estancamiento_manana",
]

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"No se encontró stations.yaml en: {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    stations_config = yaml.safe_load(f)["stations"]

now_utc = datetime.now(ZoneInfo("UTC"))
resultados = []

print("=" * 105)
print("  PROTOTIPO EXPERIMENTAL DE ALERTA TEMPRANA PM2.5 — INFERENCIA DIARIA (GFS)")
print("=" * 105)

# ------------------------------------------------------------------------------
# 2. INFERENCIA POR CIUDAD
# ------------------------------------------------------------------------------
for st in stations_config:
    ciudad = st["city"]
    lat = st["latitude"]
    lon = st["longitude"]
    tz_str = st.get("timezone", "America/Santiago")

    tz_local = ZoneInfo("America/Punta_Arenas") if tz_str == "America/Aysen_Dynamic" else ZoneInfo(tz_str)
    local_now = now_utc.astimezone(tz_local)
    fecha_hoy = local_now.date()
    fecha_manana = fecha_hoy + pd.Timedelta(days=1)

    slug_ciudad = "aysen" if ciudad.lower() in ["aysén", "aysen"] else ciudad.lower()
    modelo_path = MODELS_DIR / f"alert_model_{slug_ciudad}_{TAG_MODELO}.joblib"

    if not modelo_path.exists():
        logger.error(f"[{ciudad}] No se encontró el artefacto: {modelo_path.name}. Se omite.")
        continue

    artefacto = joblib.load(modelo_path)

    # Un modelo de otra versión usaría otras definiciones de variables: se rechaza.
    if artefacto.get("tag_version") != TAG_MODELO:
        logger.error(
            f"[{ciudad}] El artefacto es de la versión '{artefacto.get('tag_version')}' y se "
            f"esperaba '{TAG_MODELO}'. Reentrena y guarda los modelos. Se omite."
        )
        continue

    modelo = artefacto.get("modelo") if artefacto.get("modelo") is not None else artefacto.get("model")
    umbral = artefacto.get("decision_threshold", artefacto.get("umbral", 0.5))
    target_pm25 = artefacto.get("target_umbral_pm25", 50.0)
    feat_cols = artefacto.get("features", FEATURES_ESPERADAS)
    modelo_actualizado = datetime.fromtimestamp(modelo_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    # Consulta a Open-Meteo GFS con past_days=2 para asegurar holgura total
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["temperature_2m", "wind_speed_10m"],
        "models": "gfs_seamless",
        "past_days": 2,
        "forecast_days": 3,
        "timezone": "UTC",
    }

    try:
        resp = requests.get(url, params=params, timeout=25)
        resp.raise_for_status()
        payload = resp.json()["hourly"]
    except Exception as e:
        logger.error(f"[{ciudad}] Error al consultar Open-Meteo: {e}")
        continue

    df_fc = pd.DataFrame(payload)
    df_fc["time_utc"] = pd.to_datetime(df_fc["time"]).dt.tz_localize("UTC")
    df_fc["time_local"] = df_fc["time_utc"].dt.tz_convert(tz_local)
    df_fc["date_local"] = df_fc["time_local"].dt.date
    df_fc["hour_local"] = df_fc["time_local"].dt.hour

    # A. Noche de hoy (18:00 a 23:00), con las 6 horas completas como en el entrenamiento
    noche_hoy = df_fc[(df_fc["date_local"] == fecha_hoy) & (df_fc["hour_local"].between(18, 23))]
    if len(noche_hoy) < 6:
        logger.error(f"[{ciudad}] Noche de hoy incompleta ({len(noche_hoy)}/6 horas). Se omite inferencia.")
        continue

    temp_noche_hoy = noche_hoy["temperature_2m"].mean()
    wind_noche_hoy = noche_hoy["wind_speed_10m"].mean()

    # B. Día completo de mañana (t+1): 24 horas (23 en el día del cambio de horario)
    dia_manana = df_fc[df_fc["date_local"] == fecha_manana]
    if len(dia_manana) < 23:
        logger.error(f"[{ciudad}] Pronóstico de mañana incompleto ({len(dia_manana)}/24 horas). Se omite.")
        continue

    temp_min_day_fc = dia_manana["temperature_2m"].min()
    wind_mean_day_fc = dia_manana["wind_speed_10m"].mean()

    # C. Noche de mañana (18:00 a 23:00 de t+1)
    noche_manana = dia_manana[dia_manana["hour_local"].between(18, 23)]
    temp_noche_manana = noche_manana["temperature_2m"].mean()
    wind_noche_manana = noche_manana["wind_speed_10m"].mean()

    # D. Deltas homogéneos GFS vs GFS (igual que en el entrenamiento)
    delta_temp = temp_noche_manana - temp_noche_hoy
    delta_wind = wind_noche_manana - wind_noche_hoy
    estancamiento = wind_noche_manana / np.maximum(temp_noche_manana + 5.0, 1.0)

    features = {
        "temp_min_day_forecast": temp_min_day_fc,
        "wind_mean_day_forecast": wind_mean_day_fc,
        "temperature_2m_noche_forecast": temp_noche_manana,
        "wind_speed_10m_noche_forecast": wind_noche_manana,
        "delta_temp_noche": delta_temp,
        "delta_wind_noche": delta_wind,
        "estancamiento_manana": estancamiento,
    }

    try:
        X_pred = pd.DataFrame([features])[feat_cols]
    except KeyError as e:
        logger.error(f"[{ciudad}] El modelo requiere una variable que este script no calcula: {e}. Se omite.")
        continue

    probabilidad = float(modelo.predict_proba(X_pred)[:, 1][0])
    alerta_emitida = int(probabilidad >= umbral)

    resultados.append({
        "fecha_ejecucion": local_now.strftime("%Y-%m-%d %H:%M"),
        "ciudad": ciudad,
        "fecha_predicha": str(fecha_manana),
        "target_umbral": f"≥{int(target_pm25)} µg/m³",
        "probabilidad": round(probabilidad, 3),
        "umbral_operativo": round(umbral, 3),
        "alerta_emitida": "SÍ" if alerta_emitida == 1 else "NO",
        "confiable": "NO" if ciudad in CIUDADES_NO_CONFIABLES else "SÍ",
        "temp_min_manana_C": round(temp_min_day_fc, 1),
        "viento_medio_manana_kmh": round(wind_mean_day_fc, 1),
        "delta_temp_noche_C": round(delta_temp, 2),
        "delta_viento_noche_kmh": round(delta_wind, 2),
        "modelo_usado": modelo_path.name,
        "modelo_actualizado": modelo_actualizado,
    })

# ------------------------------------------------------------------------------
# 3. REPORTE EN CONSOLA Y PERSISTENCIA
# ------------------------------------------------------------------------------
df_predicciones = pd.DataFrame(resultados)

if not df_predicciones.empty:
    cols_vista = [
        "ciudad", "fecha_predicha", "target_umbral", "probabilidad",
        "umbral_operativo", "alerta_emitida", "confiable", "temp_min_manana_C",
        "viento_medio_manana_kmh", "delta_temp_noche_C",
    ]
    print(df_predicciones[cols_vista].to_string(index=False))
    for ciudad_nc, motivo in CIUDADES_NO_CONFIABLES.items():
        if ciudad_nc in set(df_predicciones["ciudad"]):
            print(f"  * {ciudad_nc}: resultado no confiable ({motivo}).")
    print("=" * 105)

    if HISTORIAL_CSV.exists():
        df_previo = pd.read_csv(HISTORIAL_CSV)
        df_acum = pd.concat([df_previo, df_predicciones], ignore_index=True)
        df_acum = df_acum.drop_duplicates(subset=["ciudad", "fecha_predicha"], keep="last")
        df_acum.to_csv(HISTORIAL_CSV, index=False)
    else:
        df_predicciones.to_csv(HISTORIAL_CSV, index=False)

    logger.info(f"Historial guardado en: {HISTORIAL_CSV}")
else:
    print("No se generaron predicciones.")
    print("=" * 105)