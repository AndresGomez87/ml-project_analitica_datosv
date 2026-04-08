"""
Aplicación Streamlit — Análisis de Clasificación: Predicción de Diabetes
Dataset: Diabetes Health Indicators (BRFSS 2015)

Ejecutar desde la raíz del proyecto:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
from pathlib import Path
from scipy.stats import chi2_contingency, kruskal
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard · Predicción de Diabetes",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Rutas ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "raw" / "dataset_clasificacion" \
           / "diabetes_binary_health_indicators_BRFSS2015.csv"
DB_PATH  = BASE_DIR / "database" / "diabetes_clasificacion.db"

# ── Carga de datos (con caché) ────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM diabetes", conn)
        conn.close()
    else:
        df = pd.read_csv(CSV_PATH)
    return df

@st.cache_data
def calcular_mutual_info(df: pd.DataFrame):
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"variable": X.columns, "mutual_info": scores}) \
             .sort_values("mutual_info", ascending=False)

# ── Clasificación de variables ────────────────────────────────────────────────
VARS_CONTINUAS  = ["BMI"]
VARS_DISCRETAS  = ["MentHlth", "PhysHlth"]
VARS_ORDINALES  = ["GenHlth", "Age", "Education", "Income"]
VARS_NOMINALES  = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

ETIQUETAS_AGE = {
    1:"18–24", 2:"25–29", 3:"30–34", 4:"35–39", 5:"40–44",
    6:"45–49", 7:"50–54", 8:"55–59", 9:"60–64", 10:"65–69",
    11:"70–74", 12:"75–79", 13:"80+"
}
ETIQUETAS_EDUC = {
    1:"Sin educación", 2:"Primaria incompleta", 3:"Primaria",
    4:"Secundaria", 5:"Técnico/Tecnólogo", 6:"Universidad"
}
ETIQUETAS_ING = {
    1:"<$10K", 2:"$10–15K", 3:"$15–20K", 4:"$20–25K",
    5:"$25–35K", 6:"$35–50K", 7:"$50–75K", 8:">$75K"
}

# ── Carga ─────────────────────────────────────────────────────────────────────
try:
    df_raw = cargar_datos()
except FileNotFoundError:
    st.error("❌ No se encontró el archivo de datos. Verifica la ruta en `src/app.py`.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Filtros
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fb/Escudo-UdeA.svg", width=80)
st.sidebar.title("🔎 Filtros")
st.sidebar.markdown("Ajusta los filtros para explorar el dataset.")

# Clase objetivo
clase_sel = st.sidebar.multiselect(
    "Clase (Diabetes_binary)",
    options=[0, 1],
    default=[0, 1],
    format_func=lambda x: "0 — Sin diabetes" if x == 0 else "1 — Diabético/Prediabético"
)

# Grupo de edad
age_min, age_max = int(df_raw["Age"].min()), int(df_raw["Age"].max())
age_range = st.sidebar.slider("Grupo de edad", age_min, age_max, (age_min, age_max))

# BMI
bmi_min, bmi_max = float(df_raw["BMI"].min()), float(df_raw["BMI"].max())
bmi_range = st.sidebar.slider("IMC (BMI)", bmi_min, bmi_max, (bmi_min, bmi_max))

# Sexo
sexo_sel = st.sidebar.multiselect(
    "Sexo", options=[0, 1], default=[0, 1],
    format_func=lambda x: "Femenino" if x == 0 else "Masculino"
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: BRFSS 2015 — CDC  \n"
                   "Proyecto: Analítica de Datos  \n"
                   "Universidad de Antioquia")

# ── Aplicar filtros ───────────────────────────────────────────────────────────
df = df_raw.copy()
if clase_sel:
    df = df[df["Diabetes_binary"].isin(clase_sel)]
if sexo_sel:
    df = df[df["Sex"].isin(sexo_sel)]
df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
df = df[(df["BMI"] >= bmi_range[0]) & (df["BMI"] <= bmi_range[1])]

# ═════════════════════════════════════════════════════════════════════════════
# ENCABEZADO
# ═════════════════════════════════════════════════════════════════════════════
st.title("🩺 Dashboard — Predicción de Diabetes")
st.markdown(
    "**Dataset:** Diabetes Health Indicators (BRFSS 2015) · "
    f"**Registros filtrados:** {len(df):,} / {len(df_raw):,}"
)
st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# KPIs
# ═════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

tasa = (df["Diabetes_binary"] == 1).mean() * 100
bmi_prom = df["BMI"].mean()
edad_prom = df["Age"].mean()
sin_acceso = (df["AnyHealthcare"] == 0).mean() * 100
riesgo_alto = (
    (df["HighBP"] == 1) & (df["HighChol"] == 1) & (df["BMI"] >= 30)
).mean() * 100

k1.metric("👥 Total registros",    f"{len(df):,}")
k2.metric("📈 Tasa de diabetes",   f"{tasa:.1f}%")
k3.metric("⚖️ IMC promedio",       f"{bmi_prom:.1f}")
k4.metric("🎂 Grupo edad prom.",   f"{edad_prom:.1f}")
k5.metric("⚠️ Sin cobertura médica", f"{sin_acceso:.1f}%")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Estadísticas descriptivas",
    "📈 Distribuciones y relaciones",
    "🔗 Correlaciones",
    "🧪 Pruebas estadísticas",
    "🛠️ Imputación de datos"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Estadísticas descriptivas
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Estadísticas descriptivas")

    tipo_var = st.radio(
        "Tipo de variables a mostrar",
        ["Numéricas / Ordinales", "Categóricas binarias"],
        horizontal=True
    )

    if tipo_var == "Numéricas / Ordinales":
        vars_desc = VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES
        desc = df[vars_desc].describe().T
        desc["mediana"]  = df[vars_desc].median()
        desc["asimetría"] = df[vars_desc].skew().round(3)
        desc = desc[["count","mean","mediana","std","min","25%","75%","max","asimetría"]]
        desc.columns = ["n","media","mediana","std","min","Q1","Q3","max","asimetría"]
        st.dataframe(desc.round(3), use_container_width=True)

    else:
        rows = []
        for var in VARS_NOMINALES:
            vc = df[var].value_counts()
            rows.append({
                "variable": var,
                "total": len(df[var].dropna()),
                "frecuencia_0": vc.get(0, 0),
                "frecuencia_1": vc.get(1, 0),
                "pct_1 (%)": round(vc.get(1, 0) / len(df[var].dropna()) * 100, 2)
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("---")
    st.subheader("Distribución de la variable objetivo")

    col1, col2 = st.columns([1, 2])
    vc = df["Diabetes_binary"].value_counts().reset_index()
    vc.columns = ["clase", "cantidad"]
    vc["etiqueta"] = vc["clase"].map({0: "Sin diabetes", 1: "Diabético/Prediabético"})

    with col1:
        fig = px.pie(
            vc, values="cantidad", names="etiqueta",
            color_discrete_sequence=["#4C9BE8", "#E86B4C"],
            hole=0.45
        )
        fig.update_traces(textposition="outside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            vc, x="etiqueta", y="cantidad",
            color="etiqueta",
            color_discrete_sequence=["#4C9BE8", "#E86B4C"],
            text="cantidad",
            title="Conteo por clase"
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Registros")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"ℹ️ **Desbalanceo:** la clase mayoritaria (sin diabetes) representa el "
        f"{(df['Diabetes_binary']==0).mean()*100:.1f}% del total filtrado. "
        "Esto implica que la exactitud (accuracy) sola no es una métrica suficiente; "
        "se recomienda usar AUC-ROC, precisión y recall."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Distribuciones y relaciones
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Distribuciones y relaciones entre variables")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribución del IMC por clase**")
        fig = px.histogram(
            df, x="BMI", color="Diabetes_binary",
            barmode="overlay", nbins=60, opacity=0.7,
            color_discrete_map={0: "#4C9BE8", 1: "#E86B4C"},
            labels={"Diabetes_binary": "Clase", "BMI": "IMC"},
        )
        fig.add_vline(x=25, line_dash="dash", line_color="gray",
                      annotation_text="Sobrepeso", annotation_position="top right")
        fig.add_vline(x=30, line_dash="dash", line_color="black",
                      annotation_text="Obeso", annotation_position="top right")
        fig.update_layout(legend_title_text="Clase")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Tasa de diabetes por grupo de edad**")
        tasa_edad = df.groupby("Age")["Diabetes_binary"].mean().reset_index()
        tasa_edad["Grupo edad"] = tasa_edad["Age"].map(ETIQUETAS_AGE)
        tasa_edad["Tasa (%)"] = (tasa_edad["Diabetes_binary"] * 100).round(2)
        fig = px.bar(
            tasa_edad, x="Grupo edad", y="Tasa (%)",
            color="Tasa (%)", color_continuous_scale="Oranges",
            title=""
        )
        fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        var_box = st.selectbox(
            "Variable para boxplot por clase",
            VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES
        )
        df_box = df.copy()
        df_box["Clase"] = df_box["Diabetes_binary"].map(
            {0: "Sin diabetes", 1: "Diabético/Prediabético"}
        )
        fig = px.box(
            df_box, x="Clase", y=var_box, color="Clase",
            color_discrete_map={"Sin diabetes": "#4C9BE8", "Diabético/Prediabético": "#E86B4C"},
            points=False, notched=True,
            title=f"Distribución de {var_box} por clase"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        var_cat = st.selectbox(
            "Variable categórica — tasa de diabetes",
            VARS_NOMINALES + VARS_ORDINALES
        )
        tasa_cat = df.groupby(var_cat)["Diabetes_binary"].mean().reset_index()
        tasa_cat.columns = [var_cat, "Tasa (%)"]
        tasa_cat["Tasa (%)"] = (tasa_cat["Tasa (%)"] * 100).round(2)
        if var_cat == "Age":
            tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_AGE)
        elif var_cat == "Education":
            tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_EDUC)
        elif var_cat == "Income":
            tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_ING)
        fig = px.bar(
            tasa_cat, x=var_cat, y="Tasa (%)",
            color="Tasa (%)", color_continuous_scale="Reds",
            title=f"Tasa de diabetes por {var_cat}"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Tasa de diabetes por nivel de ingresos y educación**")
    pivot = df.groupby(["Income", "Education"])["Diabetes_binary"].mean().reset_index()
    pivot_table = pivot.pivot(index="Income", columns="Education", values="Diabetes_binary") * 100

    fig = px.imshow(
        pivot_table.round(1),
        color_continuous_scale="RdYlGn_r",
        aspect="auto",
        labels=dict(x="Educación", y="Ingresos", color="Tasa diabetes (%)"),
        title="Tasa de diabetes (%) por nivel de ingresos y educación"
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Correlaciones
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Análisis de correlación")

    metodo_corr = st.radio(
        "Método de correlación",
        ["Spearman", "Pearson"],
        horizontal=True
    )

    corr = df.corr(method=metodo_corr.lower())

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale="RdYlGn",
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
        hoverongaps=False
    ))
    fig.update_layout(
        title=f"Matriz de correlación de {metodo_corr}",
        height=600,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Correlación de cada variable con `Diabetes_binary`**")

    target_corr = corr["Diabetes_binary"].drop("Diabetes_binary").sort_values()
    colors = ["#E86B4C" if v > 0 else "#4C9BE8" for v in target_corr.values]

    fig = go.Figure(go.Bar(
        x=target_corr.values,
        y=target_corr.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in target_corr.values],
        textposition="outside"
    ))
    fig.add_vline(x=0, line_width=1, line_color="black")
    fig.update_layout(
        title=f"Correlación de {metodo_corr} con Diabetes_binary",
        xaxis_title="Coeficiente de correlación",
        yaxis_title="",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Información Mutua con `Diabetes_binary`**")
    mi_df = calcular_mutual_info(df)

    fig = px.bar(
        mi_df.sort_values("mutual_info"),
        x="mutual_info", y="variable",
        orientation="h",
        color="mutual_info",
        color_continuous_scale="Blues",
        title="Mutual Information Score respecto a Diabetes_binary",
        labels={"mutual_info": "Mutual Info", "variable": "Variable"}
    )
    fig.update_layout(coloraxis_showscale=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Pruebas estadísticas
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Pruebas de asociación y dependencia")

    st.markdown("#### Chi-cuadrado — Variables categóricas nominales vs `Diabetes_binary`")
    st.caption("Prueba si existe asociación estadística entre cada variable y la clase objetivo.")

    @st.cache_data
    def calcular_chi2(df: pd.DataFrame):
        resultados = []
        for var in VARS_NOMINALES:
            tabla = pd.crosstab(df[var], df["Diabetes_binary"])
            chi2, p, dof, _ = chi2_contingency(tabla)
            n = tabla.sum().sum()
            cramer_v = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
            resultados.append({
                "Variable": var,
                "Chi²": round(chi2, 2),
                "p-valor": round(p, 6),
                "gl": dof,
                "V de Cramér": round(cramer_v, 4),
                "Significativa (α=0.05)": "✅ Sí" if p < 0.05 else "❌ No"
            })
        return pd.DataFrame(resultados).sort_values("V de Cramér", ascending=False)

    chi2_df = calcular_chi2(df)
    st.dataframe(chi2_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Kruskal-Wallis — Variables numéricas/ordinales vs `Diabetes_binary`")
    st.caption("Prueba si las distribuciones de las variables difieren entre las dos clases.")

    @st.cache_data
    def calcular_kruskal(df: pd.DataFrame):
        resultados = []
        for var in VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES:
            g0 = df[df["Diabetes_binary"] == 0][var].dropna()
            g1 = df[df["Diabetes_binary"] == 1][var].dropna()
            if len(g0) > 0 and len(g1) > 0:
                stat, p = kruskal(g0, g1)
                resultados.append({
                    "Variable": var,
                    "Media (clase 0)": round(g0.mean(), 3),
                    "Media (clase 1)": round(g1.mean(), 3),
                    "Estadístico H": round(stat, 2),
                    "p-valor": round(p, 8),
                    "Diferencia significativa": "✅ Sí" if p < 0.05 else "❌ No"
                })
        return pd.DataFrame(resultados).sort_values("Estadístico H", ascending=False)

    kw_df = calcular_kruskal(df)
    st.dataframe(kw_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Interpretación")
    st.info(
        "**Chi-cuadrado:** valores p < 0.05 indican asociación estadísticamente significativa. "
        "La **V de Cramér** mide el tamaño del efecto: >0.1 efecto pequeño, >0.3 moderado, >0.5 grande.  \n\n"
        "**Kruskal-Wallis:** alternativa no paramétrica al ANOVA. Valores p < 0.05 indican que las "
        "distribuciones de la variable difieren significativamente entre diabéticos y no diabéticos."
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Imputación de datos
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Tratamiento de datos faltantes e imputación")

    st.info(
        "El dataset BRFSS 2015 no contiene valores nulos en su versión original. "
        "Para demostrar y comparar los métodos de imputación, se introduce un porcentaje "
        "controlado de valores faltantes artificiales en las variables numéricas y ordinales."
    )

    missing_rate = st.slider(
        "Porcentaje de valores faltantes a introducir (%)",
        min_value=1, max_value=20, value=5, step=1
    )

    vars_imputar = VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES

    @st.cache_data
    def generar_datos_con_faltantes(df, rate, seed=42):
        np.random.seed(seed)
        df_miss = df.copy()
        for var in vars_imputar:
            idx = np.random.choice(df_miss.index, size=int(len(df_miss) * rate / 100), replace=False)
            df_miss.loc[idx, var] = np.nan
        return df_miss

    df_miss = generar_datos_con_faltantes(df, missing_rate)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Valores faltantes por variable**")
        miss_info = pd.DataFrame({
            "Variable": vars_imputar,
            "Faltantes": [df_miss[v].isnull().sum() for v in vars_imputar],
            "Porcentaje (%)": [round(df_miss[v].isnull().mean()*100, 2) for v in vars_imputar]
        })
        st.dataframe(miss_info, hide_index=True, use_container_width=True)

    with col2:
        fig = px.bar(
            miss_info, x="Variable", y="Porcentaje (%)",
            color="Porcentaje (%)", color_continuous_scale="Reds",
            title="% de valores faltantes por variable"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Comparación de métodos de imputación**")

    metodo = st.radio(
        "Método a visualizar",
        ["Mediana (tendencia central)", "KNN Imputer (k=5)"],
        horizontal=True
    )

    @st.cache_data
    def imputar(df_miss):
        imp_med = SimpleImputer(strategy="median")
        df_med  = df_miss.copy()
        df_med[vars_imputar] = imp_med.fit_transform(df_miss[vars_imputar])

        imp_knn = KNNImputer(n_neighbors=5)
        df_knn  = df_miss.copy()
        df_knn[vars_imputar] = imp_knn.fit_transform(df_miss[vars_imputar])
        return df_med, df_knn

    df_mediana, df_knn = imputar(df_miss)
    df_imputado = df_mediana if "Mediana" in metodo else df_knn
    nombre_metodo = "Mediana" if "Mediana" in metodo else "KNN"

    var_vis = st.selectbox("Variable a visualizar", vars_imputar)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.histogram(
            df[var_vis].dropna(), nbins=40,
            title=f"{var_vis} — Original",
            color_discrete_sequence=["#4C9BE8"]
        )
        fig.update_layout(xaxis_title=var_vis, yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram(
            df_imputado[var_vis], nbins=40,
            title=f"{var_vis} — Imputado ({nombre_metodo})",
            color_discrete_sequence=["#E8A84C" if nombre_metodo=="Mediana" else "#4CE8A0"]
        )
        fig.update_layout(xaxis_title=var_vis, yaxis_title="Frecuencia")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Tabla comparativa de estadísticos antes y después de la imputación**")

    comp_rows = []
    for var in vars_imputar:
        comp_rows.append({
            "Variable": var, "Método": "Original",
            "Media": round(df[var].mean(), 3),
            "Mediana": round(df[var].median(), 3),
            "Std": round(df[var].std(), 3)
        })
        comp_rows.append({
            "Variable": var, "Método": "Mediana",
            "Media": round(df_mediana[var].mean(), 3),
            "Mediana": round(df_mediana[var].median(), 3),
            "Std": round(df_mediana[var].std(), 3)
        })
        comp_rows.append({
            "Variable": var, "Método": "KNN",
            "Media": round(df_knn[var].mean(), 3),
            "Mediana": round(df_knn[var].median(), 3),
            "Std": round(df_knn[var].std(), 3)
        })

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with st.expander("📖 Justificación del método recomendado"):
        st.markdown("""
**Comparativa:**

| Criterio | Mediana | KNN |
|---|---|---|
| Preserva distribución | Parcialmente (introduce picos) | Mejor: respeta la distribución local |
| Costo computacional | Muy bajo | Moderado |
| Sensible a outliers | No | Sí (distancia euclidiana) |
| Usa relaciones entre variables | No | Sí |

**Recomendación:** Para este dataset, la imputación por **KNN** es preferible porque variables
como `BMI`, `Age` y `GenHlth` tienen correlaciones conocidas entre sí. KNN aprovecha esas
relaciones para estimar valores más realistas. La mediana es aceptable como alternativa
rápida cuando el porcentaje de nulos es bajo (<5%) y la velocidad es prioritaria.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "📌 Analítica de Datos · Universidad de Antioquia  "
    "| Dataset: BRFSS 2015 — CDC  "
    "| Fuente: kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
)
