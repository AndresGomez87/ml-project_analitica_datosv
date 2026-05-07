# =============================================================================
# LABORATORIO #4 - ANALÍTICA DE DATOS - UNIVERSIDAD DE ANTIOQUIA
# Profesor: Duván Cataño
# Tema: Pipeline completo de Regresión sobre datos de Airbnb (listings)
# Variable objetivo: price (precio por noche en USD)
# =============================================================================
#
# ¿QUÉ ES ESTE ARCHIVO?
# Este script de Python realiza todo el proceso de construir modelos de Machine
# Learning para PREDECIR el precio de alojamientos en Airbnb. Es como enseñarle
# a una computadora a estimar cuánto debería costar una habitación basándose
# en sus características (barrio, tipo de cuarto, número de reseñas, etc.).
#
# ESTRUCTURA DEL SCRIPT:
#   1. Importaciones (traer herramientas)
#   2. Carga y limpieza de datos
#   3. Preparación de datos (Pipelines)
#   4. Entrenamiento de 8 modelos
#   5. Validación cruzada (evaluación justa)
#   6. Ajuste de hiperparámetros
#   7. Evaluación final en test
#   8. Gráficas de residuos e importancia de variables
# =============================================================================


# =============================================================================
# SECCIÓN 1: IMPORTACIONES
# =============================================================================
# "Importar" en Python significa traer herramientas que otras personas ya
# programaron, para no tener que hacerlo desde cero.

import warnings
warnings.filterwarnings("ignore")  # Silencia advertencias no críticas para limpieza de output

import pandas as pd          # Para manejar tablas de datos (como Excel en Python)
import numpy as np           # Para operaciones matemáticas y matriciales
import matplotlib.pyplot as plt   # Para crear gráficas
import seaborn as sns        # Para gráficas más estéticas y estadísticas
import joblib                # Para guardar modelos entrenados en disco

# --- Herramientas de Scikit-Learn (librería principal de Machine Learning) ---
from sklearn.model_selection import (
    train_test_split,        # Divide datos en entrenamiento y prueba
    cross_validate,          # Realiza validación cruzada (evalúa modelos de forma robusta)
    KFold,                   # Define la estrategia de K particiones para validación cruzada
    GridSearchCV,            # Búsqueda exhaustiva de hiperparámetros
    RandomizedSearchCV       # Búsqueda aleatoria de hiperparámetros (más rápida)
)
from sklearn.pipeline import Pipeline          # Une pasos de preprocesamiento + modelo
from sklearn.compose import ColumnTransformer  # Aplica transformaciones distintas por columna
from sklearn.preprocessing import (
    StandardScaler,          # Escala variables numéricas: media=0, desviación=1
    OneHotEncoder            # Convierte categorías en columnas numéricas de 0 y 1
)
from sklearn.impute import SimpleImputer       # Rellena valores faltantes (NaN)
from sklearn.linear_model import (
    LinearRegression,        # Regresión lineal clásica (línea recta)
    Ridge,                   # Regresión lineal con penalización L2 (evita sobreajuste)
    Lasso                    # Regresión lineal con penalización L1 (puede eliminar variables)
)
from sklearn.isotonic import IsotonicRegression  # Regresión isotónica (monotónica)
from sklearn.tree import DecisionTreeRegressor   # Árbol de decisión
from sklearn.ensemble import RandomForestRegressor  # Bosque aleatorio (muchos árboles)
from sklearn.metrics import (
    mean_absolute_error,     # MAE: error absoluto promedio
    mean_squared_error,      # MSE: error cuadrático medio
    r2_score                 # R²: qué tan bien explica el modelo la variabilidad
)

# --- Modelos de boosting (muy poderosos, ganadores de muchas competencias) ---
from xgboost import XGBRegressor      # XGBoost: árboles que se construyen corrigiendo errores
from lightgbm import LGBMRegressor    # LightGBM: similar a XGBoost pero más rápido

import os  # Para manejar rutas del sistema de archivos

print("=" * 70)
print("  LABORATORIO #4 - REGRESIÓN - ANALÍTICA DE DATOS - UdeA")
print("=" * 70)


# =============================================================================
# SECCIÓN 2: CARGA Y LIMPIEZA DE DATOS
# =============================================================================

# --- 2.1 Cargar el dataset ---
# El dataset es una tabla de datos en formato CSV (valores separados por comas).
# Contiene información de anuncios de Airbnb: precio, barrio, tipo de cuarto, etc.

RUTA_DATASET = os.path.join(
    "data", "raw", "dataset_regresion_listings.csv"
)
# NOTA: Si ejecutas este script desde la carpeta raíz del proyecto, cambia la
# ruta a: "data/raw/dataset_regresion_listings.csv"

print(f"\n[1/8] Cargando dataset desde: {RUTA_DATASET}")
try:
    df = pd.read_csv(RUTA_DATASET, low_memory=False)
    print(f"      Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
except FileNotFoundError:
    raise FileNotFoundError(
        f"No se encontró el archivo en '{RUTA_DATASET}'.\n"
        "Verifica que la ruta sea correcta relativa a donde ejecutas el script."
    )

# --- 2.2 Exploración inicial rápida ---
print("\n--- Vista previa del dataset ---")
print(df.head(3).to_string())

print("\n--- Tipos de datos y valores faltantes ---")
info_df = pd.DataFrame({
    "dtype": df.dtypes,
    "nulos": df.isnull().sum(),
    "% nulos": (df.isnull().sum() / len(df) * 100).round(2)
})
print(info_df[info_df["nulos"] > 0].to_string())

# --- 2.3 Limpieza de la variable objetivo: price ---
# El precio a veces viene como texto con símbolo "$" y comas: "$1,200.00"
# Necesitamos convertirlo a número decimal.

print("\n[2/8] Limpiando variable objetivo 'price'...")

if df["price"].dtype == object:
    # Eliminar "$", "," y espacios, luego convertir a número
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[\$,\s]", "", regex=True)  # Quitar símbolos
        .replace("", np.nan)                        # Vacíos → NaN
        .astype(float)
    )

# Eliminar filas donde el precio es nulo, cero o negativo (no tiene sentido)
n_antes = len(df)
df = df[df["price"] > 0].dropna(subset=["price"])
print(f"      Filas eliminadas por precio inválido: {n_antes - len(df)}")

# Eliminar outliers extremos de precio usando el percentil 99.
# Un outlier es un valor anómalamente alto/bajo que puede confundir al modelo.
# El percentil 99 significa: solo eliminamos el 1% más caro.
p99 = df["price"].quantile(0.99)
df = df[df["price"] <= p99]
print(f"      Precio máximo después de filtrar outliers (p99): ${p99:.2f}")
print(f"      Filas restantes: {len(df)}")

# --- 2.4 Selección de variables (features) ---
# Escogemos las columnas que usaremos para predecir el precio.
# Excluimos: listing_id (es solo un ID), name/description (texto libre complejo),
# date/available (pertenecen al calendario, no al listing en sí),
# host_id (ID nominal sin utilidad predictiva directa).

VARIABLES_NUMERICAS = [
    "latitude",            # Posición geográfica (latitud)
    "longitude",           # Posición geográfica (longitud)
    "number_of_reviews",   # Total de reseñas recibidas
    "reviews_per_month",   # Velocidad de reseñas (popularidad)
    "availability_365"     # Días disponibles al año
]

VARIABLES_CATEGORICAS = [
    "neighbourhood",       # Barrio del alojamiento
    "room_type"            # Tipo: casa entera / habitación privada / compartida
]

VARIABLE_OBJETIVO = "price"

# Verificamos que todas las columnas existan en el dataset
cols_necesarias = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + [VARIABLE_OBJETIVO]
cols_faltantes = [c for c in cols_necesarias if c not in df.columns]
if cols_faltantes:
    raise ValueError(f"Columnas no encontradas en el dataset: {cols_faltantes}\n"
                     f"Columnas disponibles: {list(df.columns)}")

# Nos quedamos solo con las columnas seleccionadas
df = df[cols_necesarias].copy()
print(f"\n      Variables seleccionadas: {len(cols_necesarias) - 1} predictoras + 1 objetivo")


# =============================================================================
# SECCIÓN 3: PREPARACIÓN DE DATOS Y PIPELINES
# =============================================================================
# Un PIPELINE es una secuencia de pasos automáticos que se aplican en orden.
# Ejemplo: rellenar nulos → escalar → entrenar modelo.
# El pipeline garantiza que los pasos de preprocesamiento se apliquen
# SOLO sobre datos de entrenamiento al hacer validación cruzada, evitando
# "data leakage" (fuga de información futura hacia el pasado).

print("\n[3/8] Construyendo Pipelines de preprocesamiento...")

# --- 3.1 Separar X (predictoras) e y (objetivo) ---
X = df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS]  # Tabla de variables predictoras
y = df[VARIABLE_OBJETIVO]                            # Columna que queremos predecir

# --- 3.2 Dividir en Train (70%) y Test (30%) ---
# Train: datos con los que el modelo APRENDE
# Test: datos que el modelo NUNCA vio, para evaluar qué tan bien generaliza
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,    # 30% para prueba
    random_state=42    # Semilla para reproducibilidad (mismo resultado cada vez)
)
print(f"      Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

# --- 3.3 Preprocesador para variables numéricas ---
# Paso 1: SimpleImputer → rellena valores faltantes con la MEDIANA
#   (La mediana es el valor central; es más robusta que el promedio ante outliers)
# Paso 2: StandardScaler → escala los números para que tengan media=0 y std=1
#   Esto es necesario para modelos lineales (Ridge, LASSO) que son sensibles
#   a la magnitud de las variables. Ejemplo: latitude (~6) vs reviews (~100).
preprocesador_numerico = Pipeline(steps=[
    ("imputar",  SimpleImputer(strategy="median")),   # Rellenar NaN con mediana
    ("escalar",  StandardScaler())                    # Estandarizar escala
])

# --- 3.4 Preprocesador para variables categóricas ---
# Paso 1: SimpleImputer → rellena categorías faltantes con la más frecuente
# Paso 2: OneHotEncoder → convierte "room_type" en columnas 0/1
#   Ejemplo: room_type="Private room" → columna_private_room=1, resto=0
#   handle_unknown="ignore": si aparece una categoría nueva en test, la ignora
preprocesador_categorico = Pipeline(steps=[
    ("imputar",  SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# --- 3.5 Combinar ambos preprocesadores con ColumnTransformer ---
# Aplica el preprocesador numérico a las columnas numéricas
# y el categórico a las columnas categóricas, simultáneamente.
preprocesador = ColumnTransformer(transformers=[
    ("num", preprocesador_numerico,   VARIABLES_NUMERICAS),
    ("cat", preprocesador_categorico, VARIABLES_CATEGORICAS)
])

print("      Preprocesador construido correctamente.")


# =============================================================================
# SECCIÓN 4: DEFINICIÓN DE LOS 8 MODELOS
# =============================================================================
# Cada modelo es una "estrategia" diferente para aprender la relación entre
# las variables predictoras y el precio.

print("\n[4/8] Definiendo los 8 modelos de regresión...")

# Nota sobre IsotonicRegression: requiere una sola variable de entrada (1D).
# Para un dataset multivariado como este, no es aplicable directamente con Pipeline.
# Se documenta el motivo y se omite del pipeline general.

MODELOS = {
    # -------------------------------------------------------------------------
    # REGRESIÓN LINEAL: asume que el precio es una suma ponderada de variables.
    # Ejemplo: precio = 50*barrio + 30*tipo_habitacion + 0.5*reseñas + ...
    # Es el modelo más simple e interpretable.
    "Regresión Lineal": LinearRegression(),

    # -------------------------------------------------------------------------
    # RIDGE: igual que regresión lineal, pero agrega una PENALIZACIÓN (alpha)
    # que evita que los coeficientes se vuelvan muy grandes.
    # Esto reduce el sobreajuste (cuando el modelo "memoriza" los datos de train
    # pero falla en datos nuevos).
    "Ridge": Ridge(alpha=1.0),

    # -------------------------------------------------------------------------
    # LASSO: similar a Ridge, pero su penalización puede volver algunos
    # coeficientes exactamente CERO, eliminando variables irrelevantes.
    # Es una forma automática de selección de variables.
    "LASSO": Lasso(alpha=1.0, max_iter=5000),

    # -------------------------------------------------------------------------
    # ÁRBOL DE DECISIÓN: divide los datos haciendo preguntas binarias sucesivas.
    # Ejemplo: ¿barrio == "Manhattan"? → SI: ¿reseñas > 50? → ...
    # Es muy interpretable pero puede sobreajustarse fácilmente.
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42, max_depth=8),

    # -------------------------------------------------------------------------
    # RANDOM FOREST: construye MUCHOS árboles de decisión, cada uno entrenado
    # con una muestra aleatoria de datos y variables. La predicción final es el
    # PROMEDIO de todos los árboles. Mucho más robusto que un solo árbol.
    "Random Forest": RandomForestRegressor(
        n_estimators=100,    # Número de árboles
        random_state=42,
        n_jobs=-1            # Usa todos los núcleos del procesador
    ),

    # -------------------------------------------------------------------------
    # XGBOOST: construye árboles SECUENCIALMENTE, donde cada árbol nuevo
    # corrige los errores del árbol anterior. Es uno de los algoritmos más
    # poderosos en competencias de Machine Learning.
    "XGBoost": XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,   # Qué tan rápido aprende (menor = más cuidadoso)
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0          # Silencia los logs de XGBoost
    ),

    # -------------------------------------------------------------------------
    # LIGHTGBM: similar a XGBoost pero más eficiente en memoria y velocidad.
    # Especialmente bueno con datasets grandes y muchas variables categóricas.
    "LightGBM": LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1           # Silencia los logs de LightGBM
    ),
}

# Envolver cada modelo en un Pipeline completo: preprocesador → modelo
PIPELINES = {
    nombre: Pipeline(steps=[
        ("preprocesador", preprocesador),   # Paso 1: limpiar y transformar datos
        ("modelo", modelo)                   # Paso 2: entrenar el modelo
    ])
    for nombre, modelo in MODELOS.items()
}

print(f"      {len(PIPELINES)} modelos definidos.")
print("      NOTA: IsotonicRegression omitida (requiere entrada 1D univariada).")


# =============================================================================
# SECCIÓN 5: VALIDACIÓN CRUZADA (K-FOLD CROSS VALIDATION)
# =============================================================================
# ¿Por qué validación cruzada?
# Si evaluamos el modelo solo una vez, el resultado puede depender de qué
# datos cayeron en train y cuáles en test (azar). La validación cruzada repite
# la evaluación K veces con diferentes divisiones y promedía los resultados.
#
# K-Fold con k=5:
#   [Fold 1] ████░░░░░░  → entrena en 80%, evalúa en 20%
#   [Fold 2] ░░░░████░░  → entrena en 80%, evalúa en otro 20%
#   ...y así 5 veces. Cada observación es evaluada exactamente una vez.
#
# MÉTRICAS:
# - MAE (Mean Absolute Error): promedio de |real - predicho|. Fácil de interpretar.
#   Si MAE=30, el modelo se equivoca en promedio $30 por noche.
# - RMSE (Root Mean Squared Error): raíz del promedio de (real - predicho)².
#   Penaliza más los errores grandes. RMSE=50 es peor que RMSE=30.
# - R² (R cuadrado): proporción de variabilidad del precio explicada por el modelo.
#   R²=1 → predicción perfecta. R²=0 → el modelo no explica nada.

print("\n[5/8] Ejecutando Validación Cruzada (K=5) para los 8 modelos...")
print("      (Esto puede tomar varios minutos...)\n")

# Configuración del K-Fold
kf = KFold(
    n_splits=5,       # 5 particiones (folds)
    shuffle=True,     # Mezclar datos antes de dividir (evita sesgos de orden)
    random_state=42
)

# Diccionario para almacenar los resultados de cada modelo
resultados_cv = {}

for nombre, pipeline in PIPELINES.items():
    print(f"  → Evaluando: {nombre}...", end=" ", flush=True)

    # cross_validate ejecuta el pipeline completo K veces
    # scoring: qué métricas calcular. El signo negativo es convención de scikit-learn
    # (internamente minimiza, por eso devuelve negativos para MAE y RMSE).
    cv_resultado = cross_validate(
        pipeline,
        X_train, y_train,
        cv=kf,
        scoring={
            "MAE":  "neg_mean_absolute_error",
            "RMSE": "neg_root_mean_squared_error",
            "R2":   "r2"
        },
        return_train_score=False,
        n_jobs=-1
    )

    # Recuperamos los valores positivos (scikit-learn los devuelve negativos)
    mae_scores  = -cv_resultado["test_MAE"]
    rmse_scores = -cv_resultado["test_RMSE"]
    r2_scores   =  cv_resultado["test_R2"]

    resultados_cv[nombre] = {
        "MAE_mean":  mae_scores.mean(),
        "MAE_std":   mae_scores.std(),
        "RMSE_mean": rmse_scores.mean(),
        "RMSE_std":  rmse_scores.std(),
        "R2_mean":   r2_scores.mean(),
    }

    print(f"MAE={mae_scores.mean():.2f} | RMSE={rmse_scores.mean():.2f} | R²={r2_scores.mean():.3f}")

# --- Construir tabla comparativa ---
tabla_cv = pd.DataFrame(resultados_cv).T  # Transponer: modelos en filas
tabla_cv = tabla_cv.sort_values("RMSE_mean")  # Ordenar de mejor a peor RMSE

print("\n" + "=" * 70)
print("  TABLA COMPARATIVA - VALIDACIÓN CRUZADA (K=5)")
print("=" * 70)
print(tabla_cv.to_string(float_format="{:.4f}".format))

# ¿Qué modelo es más estable? El que tiene menor desviación estándar (std)
modelo_estable = tabla_cv["RMSE_std"].idxmin()
# ¿Cuál presenta mayor varianza? El que tiene mayor desviación estándar
modelo_variable = tabla_cv["RMSE_std"].idxmax()

print(f"\n  ✔ Modelo más ESTABLE (menor varianza en RMSE):  {modelo_estable}")
print(f"  ✗ Modelo con MAYOR VARIANZA en RMSE:            {modelo_variable}")

# Mejor modelo según RMSE promedio
mejor_modelo_cv = tabla_cv["RMSE_mean"].idxmin()
print(f"  ★ Mejor modelo por RMSE_mean:                   {mejor_modelo_cv}")


# =============================================================================
# SECCIÓN 6: AJUSTE DE HIPERPARÁMETROS (HYPERPARAMETER TUNING)
# =============================================================================
# Los "hiperparámetros" son configuraciones del modelo que NO se aprenden
# de los datos, sino que nosotros definimos antes del entrenamiento.
# Ejemplos: alpha en Ridge/LASSO, n_estimators en Random Forest.
#
# GridSearchCV: prueba TODAS las combinaciones posibles de hiperparámetros
# y evalúa cada una con validación cruzada. Devuelve la mejor combinación.

print("\n[6/8] Ajuste de Hiperparámetros...")

# --- 6.1 Modelo LINEAL: Ridge ---
print("\n  Ajustando Ridge (modelo lineal)...")

# Grilla de valores a probar para alpha (la penalización de Ridge).
# alpha pequeño ≈ regresión lineal normal | alpha grande ≈ coeficientes muy pequeños
param_grid_ridge = {
    "modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # "modelo__" es el prefijo porque el modelo está dentro del Pipeline
}

gs_ridge = GridSearchCV(
    PIPELINES["Ridge"],          # El pipeline de Ridge
    param_grid_ridge,
    cv=kf,                       # Mismo K-Fold que antes
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=0
)
gs_ridge.fit(X_train, y_train)

mejor_alpha_ridge = gs_ridge.best_params_["modelo__alpha"]
mejor_rmse_ridge  = -gs_ridge.best_score_

print(f"    Mejor alpha: {mejor_alpha_ridge}")
print(f"    RMSE CV (mejor Ridge): {mejor_rmse_ridge:.4f}")
print(f"    RMSE CV (Ridge base):  {resultados_cv['Ridge']['RMSE_mean']:.4f}")

# --- 6.2 Modelo NO LINEAL: Random Forest ---
print("\n  Ajustando Random Forest (modelo de árboles)...")

# Grilla de hiperparámetros para Random Forest:
# n_estimators: número de árboles (más árboles = más estable pero más lento)
# max_depth: profundidad máxima de cada árbol (limita la complejidad)
# min_samples_split: mínimo de muestras para dividir un nodo
param_grid_rf = {
    "modelo__n_estimators":    [50, 100, 200],
    "modelo__max_depth":       [None, 10, 20],
    "modelo__min_samples_split": [2, 5]
}

# RandomizedSearchCV: prueba solo una MUESTRA ALEATORIA de combinaciones.
# Con n_iter=20, prueba 20 combinaciones en lugar de las 18 posibles (3×3×2).
# Más eficiente cuando la grilla es grande.
gs_rf = RandomizedSearchCV(
    PIPELINES["Random Forest"],
    param_grid_rf,
    n_iter=15,               # Probar 15 combinaciones aleatorias
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=0
)
gs_rf.fit(X_train, y_train)

mejor_params_rf = gs_rf.best_params_
mejor_rmse_rf   = -gs_rf.best_score_

print(f"    Mejores hiperparámetros: {mejor_params_rf}")
print(f"    RMSE CV (mejor RF):      {mejor_rmse_rf:.4f}")
print(f"    RMSE CV (RF base):       {resultados_cv['Random Forest']['RMSE_mean']:.4f}")


# =============================================================================
# SECCIÓN 7: EVALUACIÓN FINAL - HOLD-OUT TEST SET
# =============================================================================
# Ahora evaluamos el MEJOR modelo encontrado en el conjunto de TEST.
# Este conjunto NUNCA fue visto durante el entrenamiento ni la búsqueda de
# hiperparámetros. Es la prueba definitiva de qué tan bien generaliza el modelo.

print("\n[7/8] Evaluación Final en conjunto de Test (Hold-Out)...")

# Seleccionar el mejor modelo entre todos los candidatos finales
# Comparamos: mejor Ridge ajustado, mejor RF ajustado, y el mejor modelo de CV
candidatos_finales = {
    "Ridge (ajustado)":         (gs_ridge.best_estimator_, mejor_rmse_ridge),
    "Random Forest (ajustado)": (gs_rf.best_estimator_,   mejor_rmse_rf),
}
# Añadir el mejor modelo de CV si no es Ridge ni RF
if mejor_modelo_cv not in ["Ridge", "Random Forest"]:
    pipeline_mejor_cv = PIPELINES[mejor_modelo_cv]
    pipeline_mejor_cv.fit(X_train, y_train)
    y_pred_cv_temp = pipeline_mejor_cv.predict(X_test)
    rmse_temp = np.sqrt(mean_squared_error(y_test, y_pred_cv_temp))
    candidatos_finales[f"{mejor_modelo_cv} (CV)"] = (pipeline_mejor_cv, rmse_temp)

# Elegir el candidato con menor RMSE en CV
nombre_ganador = min(candidatos_finales, key=lambda k: candidatos_finales[k][1])
modelo_final, _ = candidatos_finales[nombre_ganador]

print(f"    Modelo seleccionado: {nombre_ganador}")

# Predicciones sobre el conjunto de test
y_pred_test = modelo_final.predict(X_test)

# Calcular métricas finales sobre test
mae_test  = mean_absolute_error(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test   = r2_score(y_test, y_pred_test)

print("\n  --- RESULTADOS EN TEST (HOLD-OUT) ---")
print(f"    MAE:  ${mae_test:.2f}  → El modelo se equivoca en promedio ${mae_test:.2f} por noche")
print(f"    RMSE: ${rmse_test:.2f}  → Error cuadrático medio (penaliza errores grandes)")
print(f"    R²:   {r2_test:.4f}  → El modelo explica el {r2_test*100:.1f}% de la variabilidad del precio")

# Comparación CV vs Test
cv_rmse = candidatos_finales[nombre_ganador][1]
print(f"\n    RMSE en CV (entrenamiento): ${cv_rmse:.2f}")
print(f"    RMSE en Test (producción):  ${rmse_test:.2f}")
diferencia = rmse_test - cv_rmse
if abs(diferencia) / cv_rmse < 0.10:
    print("    ✔ El modelo GENERALIZA BIEN (diferencia < 10%)")
else:
    print(f"    ⚠ Diferencia notable: ${diferencia:.2f} → posible sobreajuste")


# =============================================================================
# SECCIÓN 8: GRÁFICAS
# =============================================================================
# Las gráficas nos ayudan a entender visualmente el comportamiento del modelo.

print("\n[8/8] Generando gráficas...")

# Calcular residuos: la diferencia entre el valor real y lo que predijo el modelo
# Residuo = real - predicho
# Un buen modelo tiene residuos pequeños y distribuidos aleatoriamente alrededor de 0.
residuos = y_test.values - y_pred_test

# Configuración general de estilo para todas las gráficas
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = "#2563EB"  # Azul institucional

# ===========================================================================
# FIGURA 1: Tabla comparativa de modelos (CV)
# ===========================================================================
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.axis("off")

# Preparar datos para la tabla visual
tabla_display = tabla_cv.copy()
tabla_display.columns = ["MAE Prom", "MAE Std", "RMSE Prom", "RMSE Std", "R² Prom"]
tabla_display = tabla_display.round(3)

tabla_visual = ax1.table(
    cellText=tabla_display.values,
    rowLabels=tabla_display.index,
    colLabels=tabla_display.columns,
    cellLoc="center",
    loc="center"
)
tabla_visual.auto_set_font_size(False)
tabla_visual.set_fontsize(9)
tabla_visual.scale(1.2, 1.8)

# Colorear la fila del mejor modelo en verde claro
for i, modelo_nombre in enumerate(tabla_display.index):
    color = "#d1fae5" if modelo_nombre == mejor_modelo_cv else "white"
    for j in range(len(tabla_display.columns)):
        tabla_visual[i + 1, j].set_facecolor(color)

ax1.set_title("Comparación de Modelos – Validación Cruzada (K=5)\n"
              "(Fila verde = mejor modelo por RMSE)", fontsize=12, pad=20)
plt.tight_layout()
plt.savefig("fig1_comparacion_modelos.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Figura 1 guardada: fig1_comparacion_modelos.png")

# ===========================================================================
# FIGURA 2: RMSE por modelo (barras comparativas)
# ===========================================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle("Métricas de Validación Cruzada por Modelo", fontsize=13, fontweight="bold")

metricas = [("RMSE_mean", "RMSE_std", "RMSE Promedio (±std)", "salmon"),
            ("MAE_mean",  "MAE_std",  "MAE Promedio (±std)",  "steelblue"),
            ("R2_mean",   None,       "R² Promedio",          "seagreen")]

for ax, (col_mean, col_std, titulo, color) in zip(axes2, metricas):
    valores = tabla_cv[col_mean]
    nombres = [n.replace(" ", "\n") for n in tabla_cv.index]  # Salto de línea en nombres
    errores = tabla_cv[col_std] if col_std else None

    barras = ax.barh(nombres, valores, xerr=errores, color=color,
                     alpha=0.8, capsize=4, edgecolor="white")
    ax.set_xlabel(titulo, fontsize=10)
    ax.set_title(titulo, fontsize=10, fontweight="bold")
    ax.invert_yaxis()  # El mejor (menor RMSE/MAE) queda arriba

plt.tight_layout()
plt.savefig("fig2_metricas_cv.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Figura 2 guardada: fig2_metricas_cv.png")

# ===========================================================================
# FIGURA 3: ANÁLISIS DE RESIDUOS
# ===========================================================================
# Los residuos revelan si el modelo comete errores sistemáticos.
# Un buen modelo tiene residuos que:
#   a) Se distribuyen aleatoriamente (sin patrones)
#   b) Tienen media ≈ 0
#   c) Son homocedásticos (varianza constante, sin "embudo")

fig3, axes3 = plt.subplots(2, 2, figsize=(13, 10))
fig3.suptitle(f"Análisis de Residuos – {nombre_ganador}", fontsize=13, fontweight="bold")

# --- Gráfica 3a: Residuos vs Valores Predichos ---
# Si hay un patrón (curva, embudo), el modelo tiene problemas.
ax = axes3[0, 0]
ax.scatter(y_pred_test, residuos, alpha=0.4, s=15, color=PALETTE, edgecolors="none")
ax.axhline(0, color="red", linewidth=1.5, linestyle="--",
           label="Residuo = 0 (predicción perfecta)")
ax.set_xlabel("Precio Predicho ($)")
ax.set_ylabel("Residuo (Real − Predicho)")
ax.set_title("Residuos vs Predicciones\n(buscar patrón aleatorio sin forma de embudo)")
ax.legend(fontsize=8)

# --- Gráfica 3b: Histograma de residuos ---
# Idealmente debería ser una campana de Gauss centrada en 0.
ax = axes3[0, 1]
ax.hist(residuos, bins=50, color=PALETTE, alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linewidth=1.5, linestyle="--")
ax.axvline(residuos.mean(), color="orange", linewidth=1.5, linestyle="-",
           label=f"Media residuos: ${residuos.mean():.2f}")
ax.set_xlabel("Residuo ($)")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de Residuos\n(idealmente: campana centrada en 0)")
ax.legend(fontsize=8)

# --- Gráfica 3c: Valores Reales vs Predichos ---
# La nube de puntos debería seguir la línea diagonal (predicción perfecta).
ax = axes3[1, 0]
lim_min = min(y_test.min(), y_pred_test.min())
lim_max = max(y_test.max(), y_pred_test.max())
ax.scatter(y_test, y_pred_test, alpha=0.4, s=15, color="seagreen", edgecolors="none")
ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--",
        linewidth=1.5, label="Predicción perfecta (y=x)")
ax.set_xlabel("Precio Real ($)")
ax.set_ylabel("Precio Predicho ($)")
ax.set_title(f"Real vs Predicho\nR² = {r2_test:.3f} | RMSE = ${rmse_test:.2f}")
ax.legend(fontsize=8)

# --- Gráfica 3d: Residuos vs Índice (orden de observación) ---
# Detecta si hay autocorrelación (los errores dependen de observaciones cercanas).
ax = axes3[1, 1]
ax.scatter(range(len(residuos)), residuos, alpha=0.3, s=10, color="purple", edgecolors="none")
ax.axhline(0, color="red", linewidth=1.5, linestyle="--")
ax.set_xlabel("Índice de Observación")
ax.set_ylabel("Residuo ($)")
ax.set_title("Residuos vs Índice\n(detectar autocorrelación o tendencia temporal)")

plt.tight_layout()
plt.savefig("fig3_analisis_residuos.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Figura 3 guardada: fig3_analisis_residuos.png")

# ===========================================================================
# FIGURA 4: IMPORTANCIA DE VARIABLES
# ===========================================================================
# ¿Cuál variable influye más en el precio?
# Esta información varía según el tipo de modelo:
# - Modelos lineales: los COEFICIENTES indican la influencia
# - Modelos de árboles: el atributo feature_importances_

fig4, axes4 = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle("Interpretabilidad del Modelo Final", fontsize=13, fontweight="bold")

# --- Obtener los nombres de las variables después de OneHotEncoding ---
# El preprocesador crea nuevas columnas (ej: "room_type_Private room").
# Necesitamos recuperar esos nombres.
try:
    # Ajustar el pipeline final si no lo hemos hecho ya
    if not hasattr(modelo_final.named_steps["preprocesador"], "transformers_"):
        modelo_final.fit(X_train, y_train)

    # Nombres de las columnas numéricas (sin cambiar)
    nombres_num = VARIABLES_NUMERICAS

    # Nombres de las columnas categóricas después de OneHotEncoding
    ohe = (modelo_final
           .named_steps["preprocesador"]
           .named_transformers_["cat"]
           .named_steps["codificar"])
    nombres_cat = ohe.get_feature_names_out(VARIABLES_CATEGORICAS).tolist()

    nombres_features = nombres_num + nombres_cat

    # --- Subfigura 4a: Coeficientes del Ridge ajustado ---
    # Los coeficientes del Ridge muestran cuánto impacta cada variable en el precio.
    # Coeficiente positivo = mayor valor → precio más alto
    # Coeficiente negativo = mayor valor → precio más bajo
    ridge_pipe = gs_ridge.best_estimator_
    if not hasattr(ridge_pipe.named_steps["preprocesador"], "transformers_"):
        ridge_pipe.fit(X_train, y_train)

    coefs_ridge = ridge_pipe.named_steps["modelo"].coef_
    n_coefs = min(len(coefs_ridge), len(nombres_features))
    coefs_ridge = coefs_ridge[:n_coefs]
    nombres_coefs = nombres_features[:n_coefs]

    # Ordenar por valor absoluto (mayor impacto primero)
    orden_ridge = np.argsort(np.abs(coefs_ridge))[-15:]  # Top 15
    colores_ridge = ["#ef4444" if c < 0 else "#22c55e" for c in coefs_ridge[orden_ridge]]

    axes4[0].barh(
        [nombres_coefs[i] for i in orden_ridge],
        coefs_ridge[orden_ridge],
        color=colores_ridge, edgecolor="white"
    )
    axes4[0].axvline(0, color="black", linewidth=1)
    axes4[0].set_title("Coeficientes Ridge (Top 15)\n"
                        "Verde=positivo (+precio), Rojo=negativo (−precio)", fontsize=9)
    axes4[0].set_xlabel("Coeficiente (impacto en precio USD)")

    # --- Subfigura 4b: Feature Importance del Random Forest ajustado ---
    # En Random Forest, la importancia mide cuánto contribuye cada variable
    # a reducir el error en los árboles. Más alto = más importante.
    rf_pipe = gs_rf.best_estimator_
    if not hasattr(rf_pipe.named_steps["preprocesador"], "transformers_"):
        rf_pipe.fit(X_train, y_train)

    importancias_rf = rf_pipe.named_steps["modelo"].feature_importances_
    n_imp = min(len(importancias_rf), len(nombres_features))
    importancias_rf = importancias_rf[:n_imp]
    nombres_imp = nombres_features[:n_imp]

    orden_rf = np.argsort(importancias_rf)[-15:]  # Top 15

    axes4[1].barh(
        [nombres_imp[i] for i in orden_rf],
        importancias_rf[orden_rf],
        color="steelblue", edgecolor="white", alpha=0.85
    )
    axes4[1].set_title("Importancia de Variables – Random Forest (Top 15)\n"
                        "Mayor valor = más influyente en el precio", fontsize=9)
    axes4[1].set_xlabel("Importancia relativa")

except Exception as e:
    print(f"    ⚠ No se pudo generar el gráfico de importancia: {e}")
    axes4[0].text(0.5, 0.5, "No disponible", ha="center", va="center",
                  transform=axes4[0].transAxes)
    axes4[1].text(0.5, 0.5, "No disponible", ha="center", va="center",
                  transform=axes4[1].transAxes)

plt.tight_layout()
plt.savefig("fig4_importancia_variables.png", dpi=150, bbox_inches="tight")
plt.show()
print("    Figura 4 guardada: fig4_importancia_variables.png")


# =============================================================================
# SECCIÓN 9: GUARDADO DEL MODELO FINAL
# =============================================================================
# Guardamos el modelo entrenado para poder usarlo en el futuro sin
# necesidad de volver a entrenar. Es como "congelar" el modelo.

print("\n[+] Guardando modelo final...")

# Crear la carpeta models/ si no existe
os.makedirs("../models", exist_ok=True)

# Guardar el pipeline completo (preprocesador + modelo)
joblib.dump(modelo_final, "../models/model_regression.joblib")

# Guardar los nombres de las variables usadas
joblib.dump(VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS, "../models/features_regression.joblib")

print("    ✔ Modelo guardado en: models/model_regression.joblib")
print("    ✔ Features guardadas en: models/features_regression.joblib")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("  RESUMEN EJECUTIVO")
print("=" * 70)
print(f"  Dataset:             {len(df)} alojamientos de Airbnb")
print(f"  Variable objetivo:   price (precio por noche en USD)")
print(f"  División:            70% Train ({len(X_train)} obs) | 30% Test ({len(X_test)} obs)")
print(f"  Validación cruzada:  K=5 Folds")
print(f"  Mejor modelo CV:     {mejor_modelo_cv} (RMSE={resultados_cv[mejor_modelo_cv]['RMSE_mean']:.2f})")
print(f"  Modelo final:        {nombre_ganador}")
print(f"  Desempeño en Test:   MAE=${mae_test:.2f} | RMSE=${rmse_test:.2f} | R²={r2_test:.4f}")
print("=" * 70)
print("\n  Script completado exitosamente.")
print("  Revisa las 4 figuras guardadas para el informe del laboratorio.")
