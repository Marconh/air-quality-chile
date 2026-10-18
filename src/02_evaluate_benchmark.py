# ==============================================================================
# TABLA MAESTRA DEFINITIVA CORREGIDA: BENCHMARK FUERA DE MUESTRA E IC 95%
# ==============================================================================
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path().resolve()
BASE_DIR = (
    CURRENT_DIR
    if (CURRENT_DIR / "data" / "processed").exists()
    else CURRENT_DIR.parent
)
PROCESSED_DIR = BASE_DIR / "data" / "processed"
gold_path = PROCESSED_DIR / "daily_modeling_gold_prueba_final.parquet"

df = pd.read_parquet(gold_path)
df["date_local"] = pd.to_datetime(df["date_local"])

FEATURES_LIN = [
    "temp_min_day_forecast",
    "wind_mean_day_forecast",
    "temperature_2m_noche_forecast",
    "wind_speed_10m_noche_forecast",
    "delta_temp_noche",
    "delta_wind_noche",
    "estancamiento_manana",
]

CONFIG = {
    "Santiago": {"desde": None, "fecha_test": "2025-01-01", "target": 50.0},
    "Temuco": {"desde": None, "fecha_test": "2025-01-01", "target": 50.0},
    "Aysén": {"desde": None, "fecha_test": "2025-01-01", "target": 80.0},
    "Valdivia": {
        "desde": "2025-04-01",
        "fecha_test": "2026-01-01",
        "target": 50.0,
    },
}


def bootstrap_metricas_completas(te_df, n_boot=500, seed=42):
  """Bootstrap por bloques semanales para intervalos IC 95% conjuntos."""
  te = te_df.reset_index(drop=True)
  y = te["y"].to_numpy()
  f = te["pred"].to_numpy()
  p = te["prob"].to_numpy()
  # CORRECCIÓN CLAVE: forzar tipo bool para indexación booleana real
  ini = te["es_primer_dia"].to_numpy().astype(bool)

  grupos = te.groupby(te["date_local"].dt.to_period("W").astype(str)).indices
  claves = list(grupos)
  rng = np.random.default_rng(seed)

  aucs, precs, lifts, recs_ini = [], [], [], []

  for _ in range(n_boot):
    elegidos = rng.choice(len(claves), size=len(claves), replace=True)
    idx = np.concatenate([grupos[claves[i]] for i in elegidos])
    yi, fi, pi, inii = y[idx], f[idx], p[idx], ini[idx]

    tasa_base_i = yi.mean()

    if len(np.unique(yi)) > 1:
      aucs.append(roc_auc_score(yi, pi))

    if fi.sum() > 0 and tasa_base_i > 0:
      prec_i = ((fi == 1) & (yi == 1)).sum() / fi.sum()
      precs.append(prec_i)
      lifts.append(prec_i / tasa_base_i)

    if inii.sum() > 0:
      recs_ini.append(fi[inii].mean())

  fmt = lambda v: (
      f"[{np.percentile(v, 2.5):.2f} - {np.percentile(v, 97.5):.2f}]"
      if len(v) >= 50
      else "N/A"
  )
  low_lift = np.percentile(lifts, 2.5) if len(lifts) >= 50 else np.nan

  return fmt(aucs), fmt(precs), fmt(lifts), fmt(recs_ini), low_lift


tabla_maestra = []

for ciudad, cfg in CONFIG.items():
  d = df[(df["city"] == ciudad) & df["month"].isin([5, 6, 7, 8])].copy()
  if cfg["desde"]:
    d = d[d["date_local"] >= cfg["desde"]]

  target_val = cfg["target"]
  d["y"] = (d["pm25_next_day"] >= target_val).astype(int)

  # Primer día estricto: dato válido previo, hoy < umbral y mañana >= umbral
  d["es_primer_dia"] = (
      d["pm25_mean_day"].notna()
      & (d["pm25_mean_day"] < target_val)
      & (d["pm25_next_day"] >= target_val)
  ).astype(int)

  d = d.dropna(subset=FEATURES_LIN + ["y"]).reset_index(drop=True)

  tr = d[d["date_local"] < cfg["fecha_test"]].copy()
  te = d[d["date_local"] >= cfg["fecha_test"]].copy()

  clf = make_pipeline(
      StandardScaler(),
      LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000),
  ).fit(tr[FEATURES_LIN], tr["y"])

  probs_tr = clf.predict_proba(tr[FEATURES_LIN])[:, 1]
  umbral_30 = float(np.quantile(probs_tr, 0.70))

  te["prob"] = clf.predict_proba(te[FEATURES_LIN])[:, 1]
  te["pred"] = (te["prob"] >= umbral_30).astype(int)

  y_true = te["y"].to_numpy()
  pred = te["pred"].to_numpy()
  n_test = len(te)

  tp = int(((pred == 1) & (y_true == 1)).sum())
  fp = int(((pred == 1) & (y_true == 0)).sum())
  n_alertas_reales = int(y_true.sum())

  n_primer_dia = int(te["es_primer_dia"].sum())
  tp_primer_dia = int(pred[te["es_primer_dia"] == 1].sum())
  rec_ini_val = (
      tp_primer_dia / n_primer_dia if n_primer_dia > 0 else np.nan
  )

  auc_val = roc_auc_score(y_true, te["prob"])
  tasa_base = n_alertas_reales / n_test
  prec_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
  lift_val = prec_val / tasa_base if tasa_base > 0 else 0.0
  ratio_fp_tp = fp / tp if tp > 0 else np.nan

  ic_auc, ic_prec, ic_lift, ic_rec_ini, low_lift = (
      bootstrap_metricas_completas(te)
  )

  if low_lift < 1.0:
    veredicto = "No distinguible del azar"
  elif low_lift >= 1.6:
    veredicto = "Cerca del doble del azar"
  else:
    veredicto = "Mejora sobre azar, incierta"

  tabla_maestra.append({
      "Ciudad": ciudad,
      "Días Test": n_test,
      "Tasa Base": f"{tasa_base:.1%}",
      "AUC ROC": f"{auc_val:.3f} {ic_auc}",
      "Precisión": f"{prec_val:.1%} {ic_prec}",
      "Lift vs Azar": f"{lift_val:.2f}x {ic_lift}",
      "1er Día (Recall)": (
          f"{tp_primer_dia}/{n_primer_dia} ({rec_ini_val:.1%}) {ic_rec_ini}"
      ),
      "FP / TP": f"{ratio_fp_tp:.1f}",
      "Veredicto": veredicto,
  })

df_maestra = pd.DataFrame(tabla_maestra)

print("\n" + "=" * 145)
print(
    "  TABLA MAESTRA CONSOLIDADA — EVALUACIÓN FUERA DE MUESTRA E IC 95% BOOTSTRAP"
)
print("=" * 145)
print(df_maestra.to_string(index=False))
print("=" * 145)

out_path = PROCESSED_DIR / "tabla_maestra_evaluacion_final.csv"
df_maestra.to_csv(out_path, index=False)
print(f"Tabla guardada en: {out_path}")