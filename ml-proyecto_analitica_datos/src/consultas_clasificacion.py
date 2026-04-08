"""
Consultas SQL — Dataset de Clasificación: Diabetes Health Indicators
Ejecutar después de src/crear_base_datos.py

Uso:
    python src/consultas_clasificacion.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "database" / "diabetes_clasificacion.db"

conn = sqlite3.connect(DB_PATH)
print(f"Conectado a: {DB_PATH}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 1 — CONTEO: total de registros y distribución de la variable objetivo
# ─────────────────────────────────────────────────────────────────────────────
q1 = """
SELECT
    Diabetes_binary                                          AS clase,
    COUNT(*)                                                 AS cantidad,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes), 2) AS porcentaje_pct
FROM diabetes
GROUP BY Diabetes_binary
ORDER BY Diabetes_binary;
"""
print("── Consulta 1: Distribución de la variable objetivo ──────────────────")
print(pd.read_sql_query(q1, conn).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 2 — PROMEDIO: estadísticos clave por clase de diabetes
# ─────────────────────────────────────────────────────────────────────────────
q2 = """
SELECT
    Diabetes_binary                  AS clase,
    ROUND(AVG(BMI), 2)               AS bmi_promedio,
    ROUND(AVG(Age), 2)               AS edad_grupo_prom,
    ROUND(AVG(GenHlth), 2)           AS salud_general_prom,
    ROUND(AVG(MentHlth), 2)          AS dias_salud_mental_prom,
    ROUND(AVG(PhysHlth), 2)          AS dias_salud_fisica_prom,
    ROUND(AVG(Income), 2)            AS ingresos_prom
FROM diabetes
GROUP BY Diabetes_binary
ORDER BY Diabetes_binary;
"""
print("\n── Consulta 2: Promedios de variables clave por clase ─────────────────")
print(pd.read_sql_query(q2, conn).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 3 — AGRUPACIÓN: tasa de diabetes por grupo de edad
# ─────────────────────────────────────────────────────────────────────────────
q3 = """
SELECT
    Age                                                         AS grupo_edad,
    COUNT(*)                                                    AS total,
    SUM(Diabetes_binary)                                        AS con_diabetes,
    ROUND(SUM(Diabetes_binary) * 100.0 / COUNT(*), 2)          AS tasa_diabetes_pct
FROM diabetes
GROUP BY Age
ORDER BY Age;
"""
print("\n── Consulta 3: Tasa de diabetes por grupo de edad ─────────────────────")
print(pd.read_sql_query(q3, conn).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 4 — FILTRO: personas no diabéticas con múltiples factores de riesgo
# ─────────────────────────────────────────────────────────────────────────────
q4 = """
SELECT
    COUNT(*) AS no_diabeticos_alto_riesgo
FROM diabetes
WHERE Diabetes_binary  = 0
  AND HighBP           = 1
  AND HighChol         = 1
  AND BMI             >= 30
  AND PhysActivity     = 0
  AND Smoker           = 1;
"""
print("\n── Consulta 4: Personas no diabéticas con 5 factores de riesgo ────────")
print(pd.read_sql_query(q4, conn).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 5 — AGRUPACIÓN: tasa de diabetes por nivel de educación e ingresos
# ─────────────────────────────────────────────────────────────────────────────
q5 = """
SELECT
    Education                                              AS nivel_educativo,
    Income                                                 AS nivel_ingresos,
    COUNT(*)                                               AS total,
    SUM(Diabetes_binary)                                   AS con_diabetes,
    ROUND(SUM(Diabetes_binary) * 100.0 / COUNT(*), 2)     AS tasa_pct
FROM diabetes
GROUP BY Education, Income
ORDER BY Education, Income;
"""
print("\n── Consulta 5: Tasa de diabetes por educación e ingresos (agrupación) ─")
print(pd.read_sql_query(q5, conn).head(20).to_string(index=False))
print("  ... (truncado a 20 filas)")

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 6 — JOIN: variables con su metadata y tasa de prevalencia
# ─────────────────────────────────────────────────────────────────────────────
q6 = """
SELECT
    m.variable,
    m.descripcion,
    m.tipo,
    ROUND(AVG(
        CASE m.variable
            WHEN 'HighBP'              THEN d.HighBP
            WHEN 'HighChol'            THEN d.HighChol
            WHEN 'Smoker'              THEN d.Smoker
            WHEN 'Stroke'              THEN d.Stroke
            WHEN 'HeartDiseaseorAttack' THEN d.HeartDiseaseorAttack
            WHEN 'PhysActivity'        THEN d.PhysActivity
            WHEN 'DiffWalk'            THEN d.DiffWalk
        END
    ) * 100, 2) AS prevalencia_pct
FROM variables_metadata m
JOIN diabetes d ON 1=1
WHERE m.tipo = 'Categórica nominal'
  AND m.variable NOT IN ('Diabetes_binary', 'Sex', 'CholCheck',
                          'Fruits', 'Veggies', 'HvyAlcoholConsump',
                          'AnyHealthcare', 'NoDocbcCost')
GROUP BY m.variable, m.descripcion, m.tipo
ORDER BY prevalencia_pct DESC;
"""
print("\n── Consulta 6: JOIN — prevalencia de factores de riesgo ───────────────")
print(pd.read_sql_query(q6, conn).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Consulta 7 — FILTRO avanzado: comparar IMC promedio por sexo y clase
# ─────────────────────────────────────────────────────────────────────────────
q7 = """
SELECT
    Sex                              AS sexo,
    Diabetes_binary                  AS clase,
    COUNT(*)                         AS total,
    ROUND(AVG(BMI), 2)               AS bmi_promedio,
    ROUND(MIN(BMI), 2)               AS bmi_min,
    ROUND(MAX(BMI), 2)               AS bmi_max
FROM diabetes
GROUP BY Sex, Diabetes_binary
ORDER BY Sex, Diabetes_binary;
"""
print("\n── Consulta 7: IMC promedio por sexo y clase de diabetes ──────────────")
print(pd.read_sql_query(q7, conn).to_string(index=False))

conn.close()
print("\n✓ Todas las consultas ejecutadas correctamente.")
