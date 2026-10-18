from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Definición robusta de rutas
CURRENT_DIR = Path().resolve()
if (CURRENT_DIR / "data" / "processed").exists():
  BASE_DIR = CURRENT_DIR
elif (CURRENT_DIR.parent / "data" / "processed").exists():
  BASE_DIR = CURRENT_DIR.parent
else:
  BASE_DIR = CURRENT_DIR

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Cargar el dataset Gold generado por el ETL final
GOLD_FILENAME = "daily_modeling_gold_prueba_final.parquet"
gold_path = PROCESSED_DIR / GOLD_FILENAME

if not gold_path.exists():
  raise FileNotFoundError(
      f"No se encontró el archivo Gold en: {gold_path}\n"
      "Asegúrate de haber ejecutado el nuevo ETL con OUTPUT_TAG = 'prueba_final'."
  )

df_gold = pd.read_parquet(gold_path)
df_gold["date_local"] = pd.to_datetime(df_gold["date_local"])

# 3. Parámetros del modelo de producción
TAG_MODELO = "v2"  # Mantenemos v2 como versión oficial de los artefactos
MESES_GEC = [5, 6, 7, 8]  # Ventana invernal
BUDGET_NOMINAL = 0.30  # Presupuesto parejo de alertas (30% de mayor riesgo)
UMBRAL_TARGET_DEFECTO = 50.0

CONFIG_CIUDADES = {
    "Santiago": {"desde": None, "target": 50.0},
    "Temuco": {"desde": None, "target": 50.0},
    "Aysén": {"desde": None, "target": 80.0},  # Norma y dinámica local
    "Valdivia": {
        "desde": "2025-04-01",
        "target": 50.0,
    },  # Serie post-quiebre instrumental
}

# Features físicas homogéneas (GFS vs GFS)
FEATURES_LIN = [
    "temp_min_day_forecast",
    "wind_mean_day_forecast",
    "temperature_2m_noche_forecast",
    "wind_speed_10m_noche_forecast",
    "delta_temp_noche",
    "delta_wind_noche",
    "estancamiento_manana",
]


def make_logit():
  return make_pipeline(
      StandardScaler(),
      LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000),
  )


print("=" * 80)
print(f"  SERIALIZACIÓN DE MODELOS DE PRODUCCIÓN (FUENTE: {GOLD_FILENAME})")
print("=" * 80)

for ciudad, cfg in CONFIG_CIUDADES.items():
  # Filtrar por cuenca, periodo válido y meses invernales
  d = df_gold[
      (df_gold["city"] == ciudad) & df_gold["month"].isin(MESES_GEC)
  ].copy()
  if cfg["desde"]:
    d = d[d["date_local"] >= cfg["desde"]]

  target_val = cfg["target"]
  d["y"] = (d["pm25_next_day"] >= target_val).astype(int)
  d = d.dropna(subset=FEATURES_LIN + ["y"]).reset_index(drop=True)

  # Ajuste sobre el 100% de la serie histórica disponible
  clf = make_logit().fit(d[FEATURES_LIN], d["y"])

  # Calibración del corte de probabilidad al percentil 70 (top 30% de riesgo)
  probs = clf.predict_proba(d[FEATURES_LIN])[:, 1]
  decision_threshold = float(np.quantile(probs, 1.0 - BUDGET_NOMINAL))

  # Empaquetado estandarizado
  payload = {
      "ciudad": ciudad,
      "tipo_estrategia": "modelo_puro_meteorologico_gfs",
      "modelo": clf,
      "model": clf,  # Doble llave para evitar KeyErrors
      "features": FEATURES_LIN,
      "decision_threshold": decision_threshold,
      "umbral": decision_threshold,  # Doble llave para evitar KeyErrors
      "target_umbral_pm25": target_val,
      "budget_nominal": BUDGET_NOMINAL,
      "training_window": (
          f"{d['date_local'].min().date()} a {d['date_local'].max().date()}"
      ),
      "n_training_days": len(d),
      "gold_source": GOLD_FILENAME,
      "tag_version": TAG_MODELO,
  }

  slug = "aysen" if ciudad.lower() in ["aysén", "aysen"] else ciudad.lower()
  filename = f"alert_model_{slug}_{TAG_MODELO}.joblib"
  filepath = MODELS_DIR / filename

  joblib.dump(payload, filepath)

  print(f"\n[{ciudad.upper()}]")
  print(f"  • Archivo guardado : {filepath.name}")
  print(
      f"  • Rango histórico  : {payload['training_window']} ({payload['n_training_days']} días invernales)"
  )
  print(f"  • Target evaluado  : ≥ {int(target_val)} µg/m³")
  print(
      f"  • Umbral operativo : {decision_threshold:.3f} (Top {int(BUDGET_NOMINAL*100)}% de riesgo)"
  )

print("\n" + "=" * 80)
print(f"✓ Modelos sincronizados exitosamente en: {MODELS_DIR}")
print("=" * 80)