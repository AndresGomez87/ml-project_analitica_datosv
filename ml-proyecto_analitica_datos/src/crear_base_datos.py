"""
Script para crear la base de datos SQLite e importar
el dataset de clasificación de diabetes (BRFSS 2015).

Estructura esperada del repositorio:
    ml-proyecto_analítica_datos/
        data/raw/dataset_clasificacion/
            diabetes_binary_health_indicators_BRFSS2015.csv
        database/
            diabetes_clasificacion.db   ← se crea aquí
"""

import sqlite3
import pandas as pd
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent          # raíz del proyecto
CSV_PATH   = BASE_DIR / "data" / "raw" / "dataset_clasificacion" \
             / "diabetes_binary_health_indicators_BRFSS2015.csv"
DB_PATH    = BASE_DIR / "database" / "diabetes_clasificacion.db"

# ── Cargar CSV ────────────────────────────────────────────────────────────────
print(f"Cargando CSV desde:\n  {CSV_PATH}\n")
df = pd.read_csv(CSV_PATH)
print(f"  → {df.shape[0]:,} filas  |  {df.shape[1]} columnas")

# ── Conectar / crear la base de datos ─────────────────────────────────────────
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
print(f"\nBase de datos creada/abierta en:\n  {DB_PATH}")

# ── Crear tabla principal ─────────────────────────────────────────────────────
df.to_sql("diabetes", conn, if_exists="replace", index=False)
print(f"\nTabla 'diabetes' creada con {len(df):,} registros.")

# ── Crear tabla de metadatos de variables ────────────────────────────────────
metadata = pd.DataFrame({
    "variable": [
        "Diabetes_binary", "HighBP", "HighChol", "CholCheck", "BMI",
        "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
        "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare",
        "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk",
        "Sex", "Age", "Education", "Income"
    ],
    "descripcion": [
        "Variable objetivo: 0=sin diabetes, 1=diabético/prediabético",
        "Presión arterial alta diagnosticada (0=No, 1=Sí)",
        "Colesterol alto diagnosticado (0=No, 1=Sí)",
        "Revisión de colesterol en los últimos 5 años (0=No, 1=Sí)",
        "Índice de Masa Corporal (continuo)",
        "Ha fumado al menos 100 cigarrillos en su vida (0=No, 1=Sí)",
        "Ha tenido un derrame cerebral (0=No, 1=Sí)",
        "Enfermedad coronaria o infarto al miocardio (0=No, 1=Sí)",
        "Actividad física en los últimos 30 días (0=No, 1=Sí)",
        "Consume frutas al menos una vez al día (0=No, 1=Sí)",
        "Consume verduras al menos una vez al día (0=No, 1=Sí)",
        "Consumo elevado de alcohol (0=No, 1=Sí)",
        "Tiene algún tipo de cobertura de salud (0=No, 1=Sí)",
        "No pudo ir al médico por costo en el último año (0=No, 1=Sí)",
        "Salud general autoreportada del 1 (excelente) al 5 (deficiente)",
        "Días de mala salud mental en los últimos 30 días (0-30)",
        "Días de mala salud física en los últimos 30 días (0-30)",
        "Dificultad para caminar o subir escaleras (0=No, 1=Sí)",
        "Sexo biológico (0=Femenino, 1=Masculino)",
        "Grupo de edad: 1=18-24, 2=25-29, ..., 13=80+",
        "Nivel educativo del 1 (ninguno) al 6 (universidad completa)",
        "Nivel de ingresos del 1 (<$10K) al 8 (>$75K)"
    ],
    "tipo": [
        "Categórica nominal", "Categórica nominal", "Categórica nominal",
        "Categórica nominal", "Numérica continua",
        "Categórica nominal", "Categórica nominal", "Categórica nominal",
        "Categórica nominal", "Categórica nominal", "Categórica nominal",
        "Categórica nominal", "Categórica nominal", "Categórica nominal",
        "Categórica ordinal", "Numérica discreta", "Numérica discreta",
        "Categórica nominal", "Categórica nominal",
        "Categórica ordinal", "Categórica ordinal", "Categórica ordinal"
    ]
})
metadata.to_sql("variables_metadata", conn, if_exists="replace", index=False)
print("Tabla 'variables_metadata' creada.")

# ── 5 consultas SQL relevantes ────────────────────────────────────────────────
print("\n" + "="*60)
print("CONSULTAS SQL DE VERIFICACIÓN")
print("="*60)

consultas = {
    "1. Conteo total de registros": """
        SELECT COUNT(*) AS total_registros FROM diabetes;
    """,
    "2. Distribución de la variable objetivo": """
        SELECT
            Diabetes_binary,
            COUNT(*) AS cantidad,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes), 2) AS porcentaje
        FROM diabetes
        GROUP BY Diabetes_binary
        ORDER BY Diabetes_binary;
    """,
    "3. Promedio de BMI e indicadores de salud por clase": """
        SELECT
            Diabetes_binary,
            ROUND(AVG(BMI), 2)        AS bmi_promedio,
            ROUND(AVG(MentHlth), 2)   AS dias_salud_mental_prom,
            ROUND(AVG(PhysHlth), 2)   AS dias_salud_fisica_prom,
            ROUND(AVG(Age), 2)        AS edad_grupo_prom
        FROM diabetes
        GROUP BY Diabetes_binary;
    """,
    "4. Prevalencia de diabetes por grupo de edad": """
        SELECT
            Age AS grupo_edad,
            COUNT(*) AS total,
            SUM(Diabetes_binary) AS con_diabetes,
            ROUND(SUM(Diabetes_binary) * 100.0 / COUNT(*), 2) AS tasa_diabetes_pct
        FROM diabetes
        GROUP BY Age
        ORDER BY Age;
    """,
    "5. Filtro: personas con alto riesgo acumulado": """
        SELECT COUNT(*) AS alto_riesgo
        FROM diabetes
        WHERE HighBP = 1
          AND HighChol = 1
          AND BMI >= 30
          AND PhysActivity = 0
          AND Diabetes_binary = 0;
    """,
    "6. Prevalencia de diabetes según nivel de ingresos": """
        SELECT
            Income AS nivel_ingresos,
            COUNT(*) AS total,
            SUM(Diabetes_binary) AS con_diabetes,
            ROUND(SUM(Diabetes_binary) * 100.0 / COUNT(*), 2) AS tasa_pct
        FROM diabetes
        GROUP BY Income
        ORDER BY Income;
    """,
    "7. JOIN con tabla de metadatos (ejemplo ilustrativo)": """
        SELECT
            m.variable,
            m.tipo,
            ROUND(AVG(CAST(d.Diabetes_binary AS FLOAT)), 4) AS tasa_diabetes
        FROM variables_metadata m
        JOIN diabetes d ON 1=1
        WHERE m.variable = 'Diabetes_binary'
        GROUP BY m.variable, m.tipo
        LIMIT 1;
    """
}

for titulo, sql in consultas.items():
    print(f"\n{'─'*50}")
    print(f" {titulo}")
    print(f"{'─'*50}")
    result = pd.read_sql_query(sql, conn)
    print(result.to_string(index=False))

conn.close()
print("\n✓ Base de datos creada y consultas ejecutadas correctamente.")
print(f"  Archivo: {DB_PATH}")
