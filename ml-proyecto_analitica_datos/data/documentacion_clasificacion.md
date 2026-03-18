# Documentación: Diabetes Health Indicators Dataset

## a) Nombre de la base de datos
Diabetes Health Indicators Dataset (BRFSS 2015)

## b) Fuente (URL)
- **Kaggle:** https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
- **Fuente original:** CDC — Behavioral Risk Factor Surveillance System (BRFSS), encuesta 2015

## c) Descripción general del problema
El dataset contiene 253,680 respuestas a la encuesta BRFSS 2015 del CDC (Centers for Disease Control and Prevention). La encuesta se realiza anualmente desde 1984 y recopila información sobre comportamientos de riesgo en salud, enfermedades crónicas y uso de medidas preventivas en la población adulta de Estados Unidos. El problema central es la detección temprana de diabetes y prediabetes a partir de indicadores de salud, estilo de vida y características socioeconómicas.

## d) Objetivo del análisis
Construir un modelo de clasificación que prediga si un individuo tiene diabetes o prediabetes a partir de sus indicadores de salud, estilo de vida y características demográficas. Adicionalmente, identificar cuáles variables tienen mayor poder predictivo sobre la enfermedad.

## e) Variable objetivo
`Diabetes_binary`: variable binaria que indica si el paciente tiene diabetes.
- `0` = sin diabetes
- `1` = prediabetes o diabetes

## f) Diccionario de variables

| Variable | Descripción | Tipo |
|---|---|---|
| `Diabetes_binary` | Variable objetivo: 0 = sin diabetes, 1 = diabético/prediabético | Categórica nominal |
| `HighBP` | Presión arterial alta diagnosticada (0 = No, 1 = Sí) | Categórica nominal |
| `HighChol` | Colesterol alto diagnosticado (0 = No, 1 = Sí) | Categórica nominal |
| `CholCheck` | Revisión de colesterol en los últimos 5 años (0 = No, 1 = Sí) | Categórica nominal |
| `BMI` | Índice de Masa Corporal | Numérica continua |
| `Smoker` | Ha fumado al menos 100 cigarrillos en su vida (0 = No, 1 = Sí) | Categórica nominal |
| `Stroke` | Ha tenido un derrame cerebral (0 = No, 1 = Sí) | Categórica nominal |
| `HeartDiseaseorAttack` | Enfermedad coronaria o infarto al miocardio (0 = No, 1 = Sí) | Categórica nominal |
| `PhysActivity` | Actividad física en los últimos 30 días (0 = No, 1 = Sí) | Categórica nominal |
| `Fruits` | Consume frutas al menos una vez al día (0 = No, 1 = Sí) | Categórica nominal |
| `Veggies` | Consume verduras al menos una vez al día (0 = No, 1 = Sí) | Categórica nominal |
| `HvyAlcoholConsump` | Consumo elevado de alcohol (0 = No, 1 = Sí) | Categórica nominal |
| `AnyHealthcare` | Tiene algún tipo de cobertura de salud (0 = No, 1 = Sí) | Categórica nominal |
| `NoDocbcCost` | No pudo ir al médico por costo en el último año (0 = No, 1 = Sí) | Categórica nominal |
| `GenHlth` | Salud general autoreportada del 1 (excelente) al 5 (deficiente) | Categórica ordinal |
| `MentHlth` | Días de mala salud mental en los últimos 30 días (0–30) | Numérica discreta |
| `PhysHlth` | Días de mala salud física en los últimos 30 días (0–30) | Numérica discreta |
| `DiffWalk` | Dificultad para caminar o subir escaleras (0 = No, 1 = Sí) | Categórica nominal |
| `Sex` | Sexo biológico (0 = Femenino, 1 = Masculino) | Categórica nominal |
| `Age` | Grupo de edad en rangos de 5 años, del 1 (18–24) al 13 (80+) | Categórica ordinal |
| `Education` | Nivel educativo del 1 (ninguno) al 6 (universidad completa) | Categórica ordinal |
| `Income` | Nivel de ingresos del 1 (menos de $10,000) al 8 (más de $75,000) | Categórica ordinal |

## g) Número de observaciones
253,680 registros.

## h) Número de variables
22 variables en total (21 variables predictoras + 1 variable objetivo).

> **Nota sobre desbalanceo:** el dataset presenta un desbalanceo significativo: 86.1% de casos no diabéticos y 13.9% de casos diabéticos o prediabéticos. Esto implica que la exactitud (accuracy) no es una métrica suficiente para evaluar el modelo; se recomienda usar AUC-ROC, precisión, recall y F1-score.

## i) Posibles hipótesis de estudio

1. **Hipótesis de factores fisiológicos:** Las variables de mayor importancia para predecir la diabetes son el IMC, la presión arterial alta (HighBP), el colesterol alto (HighChol), la edad (Age) y el estado de salud general (GenHlth), por encima de variables de estilo de vida como actividad física o dieta.

2. **Hipótesis de acceso a salud:** Los individuos sin cobertura médica o que no pudieron costear atención (AnyHealthcare = 0, NoDocbcCost = 1) presentan una mayor prevalencia de diabetes no diagnosticada en comparación con quienes sí tienen acceso a servicios de salud.

3. **Hipótesis de reducción dimensional:** Un modelo de clasificación entrenado con las 10 variables más correlacionadas con la variable objetivo alcanza un desempeño (AUC-ROC) comparable al modelo entrenado con las 21 variables predictoras, lo que permite simplificar el modelo sin pérdida significativa de capacidad predictiva.

4. **Hipótesis socioeconómica:** El nivel de ingresos (Income) y el nivel educativo (Education) tienen un efecto indirecto sobre el riesgo de diabetes, mediado por variables de estilo de vida como actividad física (PhysActivity), hábitos alimenticios (Fruits, Veggies) y consumo de alcohol (HvyAlcoholConsump).

## j) Limitaciones del dataset
- Los datos provienen de una encuesta de autorespuesta, por lo que algunas variables
  como GenHlth, MentHlth y PhysHlth dependen de la percepción subjetiva del encuestado.
- El dataset corresponde a datos de EE.UU. (2015), por lo que los patrones
  socioeconómicos pueden no generalizarse directamente a contextos latinoamericanos.
- El desbalanceo de clases (86% / 14%) requiere atención especial en la evaluación
  del modelo para evitar conclusiones engañosas basadas solo en exactitud.

  ## k) Metodología propuesta
1. Análisis exploratorio de datos (EDA): distribución de variables, correlaciones
   y visualización del desbalanceo de clases.
2. Preprocesamiento: verificación de valores nulos, encoding de variables
   categóricas ordinales y escalado del BMI.
3. Modelado: comparación de al menos dos clasificadores
   (por ejemplo, regresión logística y árbol de decisión o random forest).
4. Evaluación: uso de AUC-ROC, matriz de confusión, precisión y recall
   como métricas principales.
5. Interpretación: análisis de importancia de variables para responder
   las hipótesis planteadas.