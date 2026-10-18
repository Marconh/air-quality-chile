<div align="center">

# 🌬️ Air Quality Early Warning ($t+1$) · Chile
### Next-day PM2.5 episode forecasting from numerical weather forecasts only

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

[English](#english) | [Español](#español)

</div>

---

<a name="english"></a>
## English

### TL;DR
Using only numerical weather forecasts (no pollution sensor data as input), a simple logistic model flags high-risk winter days one day ahead. In **Temuco** and **Aysén** it is about twice as precise as alerting at random and catches roughly half of the first days of new episodes, something a persistence rule (*"alert tomorrow if today is an alert"*) cannot do by definition. **Valdivia** is a preliminary result (one training winter). **Santiago** does not work with these inputs. Uncertainty is wide: the test sets contain only 16–33 first days per city.

### 1. Results (Out-of-sample test, winter May–Aug)
An alert means the next-day 24-hour mean $PM_{2.5}$ is at or above $50\ \mu\text{g/m}^3$ ($80\ \mu\text{g/m}^3$ in Aysén, where 50 is exceeded on most winter days). The model alerts on the top $\approx 30\%$ of days by risk, using a threshold fixed on the training data.

| City | Test days | Base rate | AUC [95% CI] | Precision [95% CI] | Lift vs. random [95% CI] | First-day recall [95% CI] | False alarms per hit |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Santiago** | 241 | 17.4% | 0.617 [0.51 – 0.72] | 22.2% [11 – 34] | 1.28x [0.90 – 1.72] | 5/16 = 31.2% [12 – 48] | 3.5 |
| **Temuco** | 230 | 36.5% | 0.851 [0.78 – 0.91] | 77.0% [64 – 88] | 2.11x [1.70 – 2.68] | 15/26 = 57.7% [38 – 75] | 0.3 |
| **Aysén** ($\ge 80\ \mu\text{g/m}^3$) | 246 | 39.0% | 0.883 [0.84 – 0.93] | 80.7% [68 – 90] | 2.07x [1.68 – 2.82] | 18/33 = 54.5% [37 – 74] | 0.2 |
| **Valdivia** *(preliminary)* | 123 | 31.7% | 0.746 [0.63 – 0.85] | 59.4% [39 – 76] | 1.87x [1.38 – 2.50] | 9/17 = 52.9% [31 – 74] | 0.7 |

#### How to read it:
* **Base rate** is the share of days that really are alerts: what random alerting would achieve in precision. **Lift** is precision divided by base rate.
* **First day** = today is below the threshold (with valid data) and tomorrow is an alert. These are the episode onsets that persistence misses.
* **Santiago:** the lift interval includes 1, so the model is not statistically distinguishable from random there.
* Intervals are 95% percentile intervals from a block bootstrap over calendar weeks (500 resamples).

### 2. Method
* **Data:** Hourly $PM_{2.5}$ from the national air quality network (SINCA); hourly weather from Open-Meteo: ERA5 reanalysis (observations) and archived GFS Seamless forecasts. Daily $PM_{2.5}$ is accepted only with at least 18 valid hours.
* **Inputs (7, weather only):** Forecast minimum temperature and mean wind for tomorrow; forecast evening (18–23 h local) temperature and wind for tomorrow; night-to-night changes in both; and a stagnation index (evening wind divided by evening temperature + 5). No pollution variable is used as an input.
* **Night-to-night changes** are computed GFS-to-GFS (tomorrow's forecast evening minus today's forecast evening) in both training and live inference, so the pipeline does not depend on ERA5's publication delay.
* **Model:** Standardized logistic regression ($C=0.5$, balanced class weights), one per city, trained on winter days only.
* **Split:** Train 2020–2024 and test from 2025 onward (Santiago, Temuco, Aysén). Valdivia: train on 2025, test on 2026 (see below).
* **Decision rule:** Alert when the predicted probability is above the 70th percentile of training-set probabilities. The realized share of alerted days on the test set was 26–36%, not exactly 30%.
* **Final models** (used for daily inference) are refit on all available data. The metrics above come from the out-of-sample split only.

### 3. Data Engineering Notes
* **Time zones:** Aysén's switch from `America/Santiago` to a permanent UTC−3 offset in April 2025 is handled explicitly.
* **Midnight records:** SINCA's 00:00 entry that closes the previous day is shifted by one day.
* **Valdivia level shift:** The $PM_{2.5}$ series changes level from April 2025 onward ($\approx 3\times$ higher winter means, while Temuco stays stable). The cause is unconfirmed. Valdivia is therefore trained only on data from that date.
* **Forecast quality:** Mean absolute difference between the archived GFS forecast and ERA5 is 1.4–2.2 °C (temperature) and 1.8–4.5 km/h (wind) depending on the city.

### 4. Limitations
* **Few events:** First-day recall is based on 16–33 cases per city, hence the wide intervals. Two test winters (2025, 2026) is a short evaluation period.
* **Choices made on the same data:** The model family, the $\approx 30\%$ alert budget and the Aysén threshold (80 instead of 50) were chosen after exploring alternatives on this dataset.
* **Valdivia** uses a single training winter, and 2026 data are mostly preliminary (they may be revised by the network).
* **Santiago** is not usable with weather forecasts alone; it likely needs thermal inversion information that these inputs do not capture.
* **Archived vs. live forecasts:** Models are trained on archived GFS forecasts; live GFS forecasts at 24–48 h may have larger errors. Live performance has not been measured yet; the daily script keeps a prediction log for that purpose.
* **Definition of "alert":** Here it refers to a threshold on the next-day 24-hour mean (50 or 80 $\mu\text{g/m}^3$), not necessarily an official regulatory category.
* **Persistence is not replaced:** The model's value is early detection of onsets from weather alone. As an overall detector it is not uniformly better than persistence (for example, persistence can have higher total recall in some cities).

### 5. Reproduce
1. Install dependencies: `pip install -r requirements.txt`.
2. Place the SINCA $PM_{2.5}$ CSV exports in `Datos Sinca/` and configure `stations.yaml`.
3. Run the ETL: `python src/01_build_gold_dataset.py`.
4. Run the benchmark evaluation: `python src/02_evaluate_benchmark.py` (or execute the notebook).
5. Train production models: `python src/03_train_production_models.py`.
6. Daily inference (experimental): `python src/04_predict_tomorrow.py`.

---

<a name="español"></a>
## Español

### Resumen
Usando solo pronósticos meteorológicos numéricos (sin datos de contaminación como entrada), un modelo logístico simple marca con un día de anticipación los días de alto riesgo en invierno. En **Temuco** y **Aysén** es unas dos veces más preciso que alertar al azar y detecta cerca de la mitad de los primeros días de nuevos episodios, algo que una regla de persistencia (*"alerta mañana si hoy hay alerta"*) no puede hacer por definición. **Valdivia** es un resultado preliminar (un invierno de entrenamiento). **Santiago** no funciona con estas variables. La incertidumbre es amplia: el test tiene solo 16 a 33 primeros días por ciudad.

### 1. Resultados (Test fuera de muestra, invierno mayo–agosto)
Una alerta significa que el promedio de 24 horas de $PM_{2.5}$ del día siguiente es igual o mayor a $50\ \mu\text{g/m}^3$ ($80\ \mu\text{g/m}^3$ en Aysén, donde 50 se supera la mayoría de los días de invierno). El modelo alerta el $\approx 30\%$ de los días de mayor riesgo, con un umbral fijado en los datos de entrenamiento.

| Ciudad | Días de test | Tasa base | AUC [IC 95%] | Precisión [IC 95%] | Lift vs. azar [IC 95%] | Recall del primer día [IC 95%] | Falsas alarmas por acierto |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Santiago** | 241 | 17.4% | 0.617 [0.51 – 0.72] | 22.2% [11 – 34] | 1.28x [0.90 – 1.72] | 5/16 = 31.2% [12 – 48] | 3.5 |
| **Temuco** | 230 | 36.5% | 0.851 [0.78 – 0.91] | 77.0% [64 – 88] | 2.11x [1.70 – 2.68] | 15/26 = 57.7% [38 – 75] | 0.3 |
| **Aysén** ($\ge 80\ \mu\text{g/m}^3$) | 246 | 39.0% | 0.883 [0.84 – 0.93] | 80.7% [68 – 90] | 2.07x [1.68 – 2.82] | 18/33 = 54.5% [37 – 74] | 0.2 |
| **Valdivia** *(preliminar)* | 123 | 31.7% | 0.746 [0.63 – 0.85] | 59.4% [39 – 76] | 1.87x [1.38 – 2.50] | 9/17 = 52.9% [31 – 74] | 0.7 |

#### Cómo leerla:
* **Tasa base:** Porcentaje de días que realmente son alerta, es decir, la precisión que lograría alertar al azar. **Lift:** Precisión dividida por la tasa base.
* **Primer día:** Hoy está bajo el umbral (con dato válido) y mañana es alerta. Son los inicios de episodio que la persistencia no detecta.
* **Santiago:** El intervalo del lift incluye 1, así que el modelo no se distingue estadísticamente del azar.
* Los intervalos son percentiles del 95% obtenidos con bootstrap por bloques semanales (500 remuestreos).

### 2. Método
* **Datos:** $PM_{2.5}$ horario de la red nacional de calidad del aire (SINCA); clima horario de Open-Meteo: reanálisis ERA5 (observaciones) y pronósticos GFS Seamless archivados. El promedio diario de $PM_{2.5}$ se acepta solo con al menos 18 horas válidas.
* **Variables (7, solo meteorología):** Temperatura mínima y viento medio pronosticados para mañana; temperatura y viento pronosticados de la tarde-noche de mañana (18 a 23 h locales); cambios de noche a noche en ambos; y un índice de estancamiento (viento de la noche dividido por temperatura de la noche + 5). Ninguna variable de contaminación entra al modelo.
* **Cambios de noche a noche:** Se calculan GFS contra GFS (noche de mañana pronosticada menos noche de hoy pronosticada) tanto en el entrenamiento como en la inferencia en vivo, para no depender del retraso de publicación de ERA5.
* **Modelo:** Regresión logística estandarizada ($C=0.5$, pesos de clase balanceados), una por ciudad, entrenada solo con días de invierno.
* **Partición:** Entrenamiento 2020–2024 y test desde 2025 (Santiago, Temuco, Aysén). Valdivia: entrenamiento 2025 y test 2026 (ver abajo).
* **Regla de decisión:** Se alerta cuando la probabilidad supera el percentil 70 de las probabilidades del entrenamiento. El porcentaje real de días alertados en el test fue 26–36%, no exactamente 30%.
* **Modelos finales** (para la inferencia diaria): Se reentrenan con todos los datos disponibles. Las métricas de arriba provienen solo de la partición fuera de muestra.

### 3. Notas de Ingeniería de Datos
* **Zonas horarias:** Se maneja explícitamente el cambio de Aysén desde `America/Santiago` a un desfase permanente UTC−3 en abril de 2025.
* **Registros de medianoche:** La hora 00:00 de SINCA que cierra el día anterior se desplaza un día.
* **Cambio de nivel en Valdivia:** La serie de $PM_{2.5}$ cambia de nivel desde abril de 2025 (promedios de invierno unas 3 veces mayores, mientras Temuco se mantiene estable). La causa no está confirmada. Por eso Valdivia se entrena solo con datos desde esa fecha.
* **Calidad del pronóstico:** La diferencia absoluta media entre el pronóstico GFS archivado y ERA5 es de 1.4–2.2 °C (temperatura) y 1.8–4.5 km/h (viento), según la ciudad.

### 4. Limitaciones
* **Pocos eventos:** El recall del primer día se basa en 16 a 33 casos por ciudad, de ahí los intervalos anchos. Dos inviernos de test (2025 y 2026) es un periodo de evaluación corto.
* **Decisiones tomadas con los mismos datos:** La familia de modelo, el presupuesto de $\approx 30\%$ de alertas y el umbral de Aysén (80 en vez de 50) se eligieron después de probar alternativas sobre este conjunto.
* **Valdivia** usa un solo invierno de entrenamiento y los datos de 2026 son en su mayoría preliminares (la red puede revisarlos).
* **Santiago** no es utilizable solo con pronóstico meteorológico; probablemente requiere información de inversión térmica que estas variables no capturan.
* **Pronósticos archivados vs. en vivo:** Los modelos se entrenan con pronósticos GFS archivados; los pronósticos en vivo a 24–48 h pueden tener más error. El desempeño en vivo aún no está medido; el script diario guarda un historial de predicciones para eso.
* **Definición de "alerta":** Aquí es un umbral sobre el promedio de 24 horas del día siguiente (50 u 80 $\mu\text{g/m}^3$), no necesariamente una categoría regulatoria oficial.
* **No reemplaza la persistencia:** El valor del modelo es detectar inicios de episodio solo con meteorología. Como detector general no es uniformemente mejor que la persistencia (por ejemplo, la persistencia puede tener mayor recall total en algunas ciudades).

### 5. Reproducir
1. Instalar dependencias: `pip install -r requirements.txt`.
2. Colocar los CSV de $PM_{2.5}$ de SINCA en `Datos Sinca/` y configurar `stations.yaml`.
3. Ejecutar el ETL: `python src/01_build_gold_dataset.py`.
4. Ejecutar la evaluación del benchmark: `python src/02_evaluate_benchmark.py` (o ejecutar el notebook).
5. Entrenar modelos de producción: `python src/03_train_production_models.py`.
6. Inferencia diaria (experimental): `python src/04_predict_tomorrow.py`.
