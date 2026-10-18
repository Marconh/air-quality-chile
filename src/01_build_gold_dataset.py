import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ------------------------------------------------------------------------------
# CONFIGURACIÓN Y RUTAS
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("AirQualityETL")

CURRENT_DIR = Path().resolve()
BASE_DIR = (
    CURRENT_DIR
    if (CURRENT_DIR / "stations.yaml").exists()
    else CURRENT_DIR.parent
)

YAML_CONFIG_PATH = BASE_DIR / "stations.yaml"
SINCA_DIR = BASE_DIR / "Data_Sinca"
RAW_WEATHER_DIR = BASE_DIR / "data" / "raw" / "weather"
RAW_FORECAST_DIR = BASE_DIR / "data" / "raw" / "forecast"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
RAW_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_START_DATE = "2020-01-01"
GLOBAL_END_DATE = "2026-10-14"

# Sufijo de versión: evita reemplazar los parquet generados por la versión anterior.
OUTPUT_TAG = "prueba_final"
# Umbral de alerta sobre el promedio diario de PM2.5 (µg/m³).
UMBRAL_ALERTA = 50.0
# Mínimo de horas válidas para aceptar un promedio diario.
MIN_HORAS_DIA = 18

logger.info(f"Directorio base detectado: {BASE_DIR}")
logger.info(f"Ventana analítica: {GLOBAL_START_DATE} a {GLOBAL_END_DATE}")
logger.info(f"Versión de salida: {OUTPUT_TAG}")


# ------------------------------------------------------------------------------
# 1. FUNCIONES AUXILIARES DE PROCESAMIENTO Y DESCARGA
# ------------------------------------------------------------------------------
def parse_sinca_csv(
    filepath: Path,
    timezone: str = "America/Santiago",
    max_pm25: float = 1200.0,
) -> pd.DataFrame:
  """Lee y limpia el CSV del SINCA con soporte para Aysén Dinámico."""
  df = pd.read_csv(
      filepath, sep=";", decimal=",", encoding="latin1", low_memory=False
  )
  df.columns = [str(c).strip() for c in df.columns]

  col_fecha = next((c for c in df.columns if "FECHA" in c.upper()), None)
  col_hora = next((c for c in df.columns if "HORA" in c.upper()), None)

  if not col_fecha or not col_hora:
    raise KeyError(
        f"Estructura inválida en {filepath.name}: Faltan columnas de fecha u"
        " hora."
    )

  fecha_clean = (
      df[col_fecha]
      .astype(str)
      .str.strip()
      .str.replace(r"\.0$", "", regex=True)
      .str.zfill(6)
  )
  hora_clean = (
      df[col_hora]
      .astype(str)
      .str.strip()
      .str.replace(r"\.0$", "", regex=True)
      .str.zfill(4)
  )

  hora_num = pd.to_numeric(hora_clean, errors="coerce")
  hora_anterior = hora_num.shift(1)
  fecha_anterior = fecha_clean.shift(1)
  es_cierre_dia = (
      (hora_num == 0) & (hora_anterior == 2300) & (fecha_clean == fecha_anterior)
  )

  n_cierres = es_cierre_dia.sum()
  logger.info(
      f"[{filepath.name}] Horas '0' corregidas como cierre de día: {n_cierres}"
  )

  df["time_local"] = pd.to_datetime(
      fecha_clean + " " + hora_clean, format="%y%m%d %H%M", errors="coerce"
  )

  if n_cierres > 0:
    df.loc[es_cierre_dia, "time_local"] += pd.Timedelta(days=1)

  df = df.dropna(subset=["time_local"]).sort_values("time_local").copy()

  if timezone == "America/Aysen_Dynamic":
    fecha_corte = pd.Timestamp("2025-04-05 23:00:00")
    mask_hist = df["time_local"] < fecha_corte

    df_hist = df[mask_hist].copy()
    df_hist["time_utc"] = (
        df_hist["time_local"]
        .dt.tz_localize(
            "America/Santiago", ambiguous="NaT", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )

    df_post = df[~mask_hist].copy()
    df_post["time_utc"] = (
        df_post["time_local"]
        .dt.tz_localize(
            "America/Punta_Arenas", ambiguous="NaT", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )

    df = pd.concat([df_hist, df_post])
  else:
    df["time_utc"] = (
        df["time_local"]
        .dt.tz_localize(
            timezone, ambiguous="NaT", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )

  col_val = next(
      (
          c
          for c in df.columns
          if "validados" in c.lower() and "no" not in c.lower()
      ),
      None,
  )
  col_pre = next((c for c in df.columns if "preliminares" in c.lower()), None)

  if not col_val and not col_pre:
    raise KeyError(
        f"El archivo {filepath.name} no contiene columnas válidas de PM2.5."
    )

  df["pm2_5"] = np.nan
  df["data_status"] = "no_medido"

  if col_val:
    s_val = pd.to_numeric(df[col_val], errors="coerce")
    m_val = s_val.notna()
    df.loc[m_val, "pm2_5"] = s_val[m_val]
    df.loc[m_val, "data_status"] = "validado"

  if col_pre:
    s_pre = pd.to_numeric(df[col_pre], errors="coerce")
    m_pre = df["pm2_5"].isna() & s_pre.notna()
    df.loc[m_pre, "pm2_5"] = s_pre[m_pre]
    df.loc[m_pre, "data_status"] = "preliminar"

  invalid_mask = (df["pm2_5"] < 0) | (df["pm2_5"] > max_pm25)
  if invalid_mask.sum() > 0:
    logger.warning(
        f"[{filepath.name}] {invalid_mask.sum()} registros descartados por"
        f" límite físico (0 - {max_pm25} µg/m³)."
    )
    df.loc[invalid_mask, "pm2_5"] = np.nan
    df.loc[invalid_mask, "data_status"] = "anomalia_descartada"

  df_clean = df.dropna(subset=["time_utc"]).drop_duplicates(subset=["time_utc"])
  return (
      df_clean[["time_utc", "pm2_5", "data_status"]]
      .sort_values("time_utc")
      .reset_index(drop=True)
  )


def get_weather_session() -> requests.Session:
  """Sesión HTTP con reintentos exponenciales automáticos."""
  session = requests.Session()
  retries = Retry(
      total=5,
      backoff_factor=1.5,
      status_forcelist=[429, 500, 502, 503, 504],
      raise_on_status=True,
  )
  session.mount("https://", HTTPAdapter(max_retries=retries))
  return session


def trim_trailing_nulls(df: pd.DataFrame, col: str) -> pd.DataFrame:
  """Recorta las últimas filas sin dato (ERA5 tiene unos días de retraso)."""
  last_valid = df[col].last_valid_index()
  if last_valid is None:
    return df.iloc[0:0]
  return df.loc[:last_valid]


def fetch_weather_cached(
    station_id: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
  """Descarga o lee en caché la meteorología ERA5 observada en UTC."""
  cache_file = (
      cache_dir / f"weather_{station_id}_{start_date}_{end_date}.parquet"
  )

  if cache_file.exists():
    logger.info(
        f"[{station_id}] Leyendo meteorología desde caché ({cache_file.name})."
    )
    return pd.read_parquet(cache_file)

  url = "https://archive-api.open-meteo.com/v1/archive"
  variables = [
      "temperature_2m",
      "relative_humidity_2m",
      "dew_point_2m",
      "wind_speed_10m",
      "wind_direction_10m",
      "surface_pressure",
      "precipitation",
      "cloud_cover",
      "direct_normal_irradiance",
  ]

  params = {
      "latitude": lat,
      "longitude": lon,
      "start_date": start_date,
      "end_date": end_date,
      "hourly": variables,
      "timezone": "UTC",
  }

  session = get_weather_session()
  logger.info(f"[{station_id}] Solicitando ERA5 a Open-Meteo...")
  response = session.get(url, params=params, timeout=50)
  response.raise_for_status()

  payload = response.json()
  df_weather = pd.DataFrame(payload["hourly"])
  df_weather["time_utc"] = pd.to_datetime(df_weather["time"]).dt.tz_localize(
      "UTC"
  )
  df_weather = df_weather.drop(columns=["time"]).sort_values("time_utc")

  df_weather.to_parquet(cache_file, index=False)
  logger.info(f"[{station_id}] Meteorología guardada en caché.")
  return df_weather


def fetch_forecast_cached(
    station_id: str,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_dir: Path,
) -> pd.DataFrame:
  """Descarga o lee en caché las salidas de pronóstico numérico (GFS Seamless) en UTC."""
  cache_file = (
      cache_dir / f"forecast_gfs_{station_id}_{start_date}_{end_date}.parquet"
  )

  if cache_file.exists():
    logger.info(
        f"[{station_id}] Leyendo pronóstico GFS desde caché"
        f" ({cache_file.name})."
    )
    return pd.read_parquet(cache_file)

  url = "https://archive-api.open-meteo.com/v1/archive"
  params = {
      "latitude": lat,
      "longitude": lon,
      "start_date": start_date,
      "end_date": end_date,
      "hourly": ["temperature_2m", "wind_speed_10m", "precipitation"],
      "timezone": "UTC",
      "models": "gfs_seamless",
  }

  session = get_weather_session()
  logger.info(f"[{station_id}] Descargando pronósticos numéricos GFS...")
  response = session.get(url, params=params, timeout=60)
  response.raise_for_status()

  payload = response.json()
  df_fc = pd.DataFrame(payload["hourly"])
  df_fc["time_utc"] = pd.to_datetime(df_fc["time"]).dt.tz_localize("UTC")
  df_fc = df_fc.rename(
      columns={
          "temperature_2m": "temp_forecast",
          "wind_speed_10m": "wind_forecast",
          "precipitation": "precip_forecast",
      }
  ).drop(columns=["time"])

  df_fc = df_fc.sort_values("time_utc").reset_index(drop=True)
  df_fc.to_parquet(cache_file, index=False)
  logger.info(f"[{station_id}] Pronóstico GFS guardado en caché.")
  time.sleep(0.5)
  return df_fc


def flag_binaria(serie: pd.Series, umbral: float) -> pd.Series:
  """Bandera 0/1 que conserva <NA> donde no hay dato (NaN >= x NO es 0)."""
  return (serie >= umbral).astype("Int64").where(serie.notna())


# ------------------------------------------------------------------------------
# 2. BUCLE PRINCIPAL DE PROCESAMIENTO
# ------------------------------------------------------------------------------
with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
  config = yaml.safe_load(f)

consolidated_hourly_list = []
consolidated_daily_gold = []
forecast_quality = []

for st in config["stations"]:
  station_id = st["id"]
  station_name = st["name"]
  city = st["city"]
  lat = st["latitude"]
  lon = st["longitude"]
  tz = st.get("timezone", "America/Santiago")
  fname = st.get("filename") or f"m25_{station_id}.csv"
  csv_file = SINCA_DIR / fname

  logger.info("=" * 65)
  logger.info(f"Procesando: {station_name} ({city}) | Zona Horaria: {tz}")

  if not csv_file.exists():
    logger.error(f"Archivo {csv_file.name} no encontrado en {SINCA_DIR}. Se omite.")
    continue

  # 1. Parsear SINCA
  df_sinca = parse_sinca_csv(csv_file, timezone=tz)

  # 2. Meteorología observada (ERA5), sin los días finales aún vacíos
  df_weather = fetch_weather_cached(
      station_id=station_id,
      lat=lat,
      lon=lon,
      start_date=GLOBAL_START_DATE,
      end_date=GLOBAL_END_DATE,
      cache_dir=RAW_WEATHER_DIR,
  )
  df_weather = trim_trailing_nulls(df_weather, "temperature_2m")

  # 3. Pronóstico numérico (GFS)
  df_forecast = fetch_forecast_cached(
      station_id=station_id,
      lat=lat,
      lon=lon,
      start_date=GLOBAL_START_DATE,
      end_date=GLOBAL_END_DATE,
      cache_dir=RAW_FORECAST_DIR,
  )

  # 4. Unir grilla climática
  weather_and_fc = pd.merge(
      df_weather, df_forecast, on="time_utc", how="inner"
  )

  # 5. Cruce con SINCA
  merged = pd.merge(df_sinca, weather_and_fc, on="time_utc", how="right")
  merged["data_status"] = merged["data_status"].fillna("no_medido")

  merged["station_id"] = station_id
  merged["station_name"] = station_name
  merged["city"] = city

  # 5b. Diagnóstico: ¿cuánto se parece el "pronóstico" a la observación ERA5?
  mae_temp = (merged["temp_forecast"] - merged["temperature_2m"]).abs().mean()
  mae_wind = (merged["wind_forecast"] - merged["wind_speed_10m"]).abs().mean()
  forecast_quality.append({
      "city": city,
      "station_id": station_id,
      "mae_temp_C": round(float(mae_temp), 3),
      "mae_wind_kmh": round(float(mae_wind), 3),
  })
  logger.info(
      f"[{city}] Error medio pronóstico vs ERA5: temp={mae_temp:.2f} °C,"
      f" viento={mae_wind:.2f} km/h"
  )

  # 6. Descomposición temporal y time_local naive
  if tz == "America/Aysen_Dynamic":
    corte_utc = pd.Timestamp("2025-04-06 02:00:00+00:00")
    mask_hist = merged["time_utc"] < corte_utc

    dt_hist = merged.loc[mask_hist, "time_utc"].dt.tz_convert(
        "America/Santiago"
    )
    dt_post = merged.loc[~mask_hist, "time_utc"].dt.tz_convert(
        "America/Punta_Arenas"
    )

    merged.loc[mask_hist, "hour_local"] = dt_hist.dt.hour
    merged.loc[~mask_hist, "hour_local"] = dt_post.dt.hour
    merged.loc[mask_hist, "month"] = dt_hist.dt.month
    merged.loc[~mask_hist, "month"] = dt_post.dt.month
    merged.loc[mask_hist, "dayofweek"] = dt_hist.dt.dayofweek
    merged.loc[~mask_hist, "dayofweek"] = dt_post.dt.dayofweek

    merged["time_local"] = pd.concat([
        dt_hist.dt.tz_localize(None),
        dt_post.dt.tz_localize(None),
    ]).sort_index()
  else:
    dt_loc = merged["time_utc"].dt.tz_convert(tz)
    merged["hour_local"] = dt_loc.dt.hour
    merged["month"] = dt_loc.dt.month
    merged["dayofweek"] = dt_loc.dt.dayofweek
    merged["time_local"] = dt_loc.dt.tz_localize(None)

  merged["hour_local"] = merged["hour_local"].astype(int)
  merged["month"] = merged["month"].astype(int)
  merged["dayofweek"] = merged["dayofweek"].astype(int)
  merged["date_local"] = merged["time_local"].dt.floor("D")

  rad = np.radians(merged["wind_direction_10m"])
  merged["wind_u"] = -merged["wind_speed_10m"] * np.sin(rad)
  merged["wind_v"] = -merged["wind_speed_10m"] * np.cos(rad)

  consolidated_hourly_list.append(merged)

  # 7. Capa Gold diaria
  noche_mask = merged["hour_local"].between(18, 23)

  obs_daily = (
      merged.groupby("date_local")
      .agg(
          pm25_horas_validas=("pm2_5", "count"),
          pm25_mean_raw=("pm2_5", "mean"),
          pm25_max_raw=("pm2_5", "max"),
          temp_min_day=("temperature_2m", "min"),
          wind_mean_day=("wind_speed_10m", "mean"),
      )
      .reset_index()
  )
  # Promedio/máximo diario solo si hay horas suficientes
  dia_valido = obs_daily["pm25_horas_validas"] >= MIN_HORAS_DIA
  obs_daily["pm25_mean_day"] = obs_daily["pm25_mean_raw"].where(dia_valido)
  obs_daily["pm25_max_day"] = obs_daily["pm25_max_raw"].where(dia_valido)
  obs_daily = obs_daily.drop(columns=["pm25_mean_raw", "pm25_max_raw"])

  obs_noche = (
      merged[noche_mask]
      .groupby("date_local")
      .agg(
          temperature_2m_noche=("temperature_2m", "mean"),
          wind_speed_10m_noche=("wind_speed_10m", "mean"),
      )
      .reset_index()
  )

  fc_daily = (
      merged.groupby("date_local")
      .agg(
          fc_temp_min=("temp_forecast", "min"),
          fc_wind_mean=("wind_forecast", "mean"),
      )
      .reset_index()
  )

  fc_noche = (
      merged[noche_mask]
      .groupby("date_local")
      .agg(
          fc_temp_noche=("temp_forecast", "mean"),
          fc_wind_noche=("wind_forecast", "mean"),
      )
      .reset_index()
  )

  gold_station = obs_daily
  for parte in (obs_noche, fc_daily, fc_noche):
    gold_station = gold_station.merge(parte, on="date_local", how="left")

  # Calendario diario completo: garantiza que shift(±1) sea exactamente ±1 día
  gold_station = gold_station.sort_values("date_local")
  calendario = pd.date_range(
      gold_station["date_local"].min(),
      gold_station["date_local"].max(),
      freq="D",
  )
  gold_station = (
      gold_station.set_index("date_local")
      .reindex(calendario)
      .rename_axis("date_local")
      .reset_index()
  )
  gold_station["month"] = gold_station["date_local"].dt.month
  gold_station["dayofweek"] = gold_station["date_local"].dt.dayofweek

  # El pronóstico de MAÑANA (t+1) para decidir HOY (t)
  gold_station["temp_min_day_forecast"] = gold_station["fc_temp_min"].shift(-1)
  gold_station["wind_mean_day_forecast"] = gold_station["fc_wind_mean"].shift(-1)
  gold_station["temperature_2m_noche_forecast"] = gold_station[
      "fc_temp_noche"
  ].shift(-1)
  gold_station["wind_speed_10m_noche_forecast"] = gold_station[
      "fc_wind_noche"
  ].shift(-1)

  # Noche de HOY según el mismo pronóstico GFS (igual que en inferencia en vivo)
  gold_station["temp_noche_hoy_gfs"] = gold_station["fc_temp_noche"]
  gold_station["wind_noche_hoy_gfs"] = gold_station["fc_wind_noche"]

  gold_station = gold_station.drop(
      columns=["fc_temp_min", "fc_wind_mean", "fc_temp_noche", "fc_wind_noche"]
  )

  # DELTAS DEL MODELO: GFS contra GFS (noche de mañana - noche de hoy)
  gold_station["delta_temp_noche"] = (
      gold_station["temperature_2m_noche_forecast"]
      - gold_station["temp_noche_hoy_gfs"]
  )
  gold_station["delta_wind_noche"] = (
      gold_station["wind_speed_10m_noche_forecast"]
      - gold_station["wind_noche_hoy_gfs"]
  )
  # Solo referencia (NO son features): contra la noche de hoy observada (ERA5)
  gold_station["delta_temp_noche_obs"] = (
      gold_station["temperature_2m_noche_forecast"]
      - gold_station["temperature_2m_noche"]
  )
  gold_station["delta_wind_noche_obs"] = (
      gold_station["wind_speed_10m_noche_forecast"]
      - gold_station["wind_speed_10m_noche"]
  )
  gold_station["estancamiento_manana"] = gold_station[
      "wind_speed_10m_noche_forecast"
  ] / np.maximum(gold_station["temperature_2m_noche_forecast"] + 5.0, 1.0)

  # Persistencia: información disponible HOY al momento de decidir
  gold_station["pm25_lag1"] = gold_station["pm25_mean_day"].shift(1)
  gold_station["pm25_lag2"] = gold_station["pm25_mean_day"].shift(2)
  gold_station["pm25_roll3"] = (
      gold_station["pm25_mean_day"].rolling(3, min_periods=2).mean()
  )
  gold_station["pm25_roll7"] = (
      gold_station["pm25_mean_day"].rolling(7, min_periods=4).mean()
  )

  # Target (conserva <NA> cuando no hay promedio diario válido)
  gold_station["pm25_next_day"] = gold_station["pm25_mean_day"].shift(-1)
  gold_station["alerta_today"] = flag_binaria(
      gold_station["pm25_mean_day"], UMBRAL_ALERTA
  )
  gold_station["alerta_next_day"] = flag_binaria(
      gold_station["pm25_next_day"], UMBRAL_ALERTA
  )

  gold_station["station_id"] = station_id
  gold_station["city"] = city
  consolidated_daily_gold.append(gold_station)

  n_obj = int(gold_station["alerta_next_day"].notna().sum())
  n_pos = int(gold_station["alerta_next_day"].fillna(0).sum())
  logger.info(
      f"[{city}] Capa Gold: {len(gold_station):,} días | target válido:"
      f" {n_obj:,} | alertas (>= {UMBRAL_ALERTA:.0f}): {n_pos:,}"
  )

# ------------------------------------------------------------------------------
# 3. EXPORTACIÓN PARQUET (con sufijo de versión)
# ------------------------------------------------------------------------------
df_final_hourly = pd.concat(consolidated_hourly_list, ignore_index=True)
hourly_parquet = (
    PROCESSED_DIR
    / f"air_quality_chile_2020_2026_consolidated_{OUTPUT_TAG}.parquet"
)
df_final_hourly.to_parquet(hourly_parquet, index=False, engine="pyarrow")
logger.info(f"Parquet Horario Consolidado: {hourly_parquet}")

df_final_gold = pd.concat(consolidated_daily_gold, ignore_index=True)
df_final_gold = df_final_gold.dropna(
    subset=["alerta_next_day", "temp_min_day_forecast"]
).reset_index(drop=True)
df_final_gold["alerta_next_day"] = df_final_gold["alerta_next_day"].astype(int)
daily_parquet = PROCESSED_DIR / f"daily_modeling_gold_{OUTPUT_TAG}.parquet"
df_final_gold.to_parquet(daily_parquet, index=False, engine="pyarrow")

logger.info(f"Parquet Diario Gold: {daily_parquet}")
logger.info(f"Dimensiones Capa Gold: {df_final_gold.shape}")

# Resumen de positivos por ciudad y calidad del pronóstico
resumen = (
    df_final_gold.groupby("city")["alerta_next_day"]
    .agg(dias="count", alertas="sum", tasa="mean")
    .round(3)
)
logger.info("Resumen target por ciudad:\n" + resumen.to_string())

df_quality = pd.DataFrame(forecast_quality)
quality_csv = PROCESSED_DIR / f"forecast_quality_{OUTPUT_TAG}.csv"
df_quality.to_csv(quality_csv, index=False)
logger.info("Calidad del pronóstico vs ERA5:\n" + df_quality.to_string(index=False))
