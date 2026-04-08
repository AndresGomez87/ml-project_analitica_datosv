import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.feature_selection import mutual_info_classif
import warnings

warnings.filterwarnings("ignore")

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Multi-Dashboard Analítico · UdeA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo CSS personalizado ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }
.main-header {
    background: linear-gradient(135deg, #FF385C 0%, #BD1E59 60%, #6A0F49 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.main-header h1 { color: white !important; font-size: 2.2rem; margin: 0; }
.kpi-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid #FF385C; box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.78rem; color: #888; text-transform: uppercase; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #1a1a1a; }
.section-title {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
    color: #1a1a1a; border-bottom: 2px solid #FF385C; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
}
[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Rutas y Constantes ───────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH_DIABETES = BASE_DIR / "database" / "diabetes_clasificacion.db"
CSV_PATH_DIABETES = BASE_DIR / "data" / "raw" / "dataset_clasificacion" / "diabetes_binary_health_indicators_BRFSS2015.csv"
CSV_NAME_AIRBNB = "dataset_regresion_listings.csv"
AIRBNB_RED = "#FF385C"
COLOR_SEQ = ["#FF385C","#BD1E59","#6A0F49","#E8684A","#F7B731","#20BF6B"]

# ── Funciones de Carga ───────────────────────────────────────
@st.cache_data
def cargar_datos_diabetes():
    if DB_PATH_DIABETES.exists():
        conn = sqlite3.connect(DB_PATH_DIABETES)
        df = pd.read_sql("SELECT * FROM diabetes", conn)
        conn.close()
    else:
        df = pd.read_csv(CSV_PATH_DIABETES)
    return df

@st.cache_data
def cargar_datos_airbnb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df = df.drop(columns=["neighbourhood_group", "license"], errors="ignore")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    return df

# ── Sidebar: Selección de Proyecto ───────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fb/Escudo-UdeA.svg", width=80)
    st.title("🚀 Proyectos Analítica")
    opcion_proyecto = st.radio(
        "Seleccione el análisis:",
        ["🩺 Clasificación: Diabetes", "🏠 Regresión: Airbnb CDMX"]
    )
    st.markdown("---")

# ════════════════════════════════════════════════════════════
# LÓGICA: CLASIFICACIÓN DIABETES
# ════════════════════════════════════════════════════════════
if opcion_proyecto == "🩺 Clasificación: Diabetes":
    # Clasificación de variables
    VARS_CONTINUAS = ["BMI"]; VARS_DISCRETAS = ["MentHlth", "PhysHlth"]
    VARS_ORDINALES = ["GenHlth", "Age", "Education", "Income"]
    VARS_NOMINALES = ["HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]
    
    df_raw = cargar_datos_diabetes()
    
    with st.sidebar:
        st.subheader("🔎 Filtros")
        clase_sel = st.multiselect("Clase", options=[0, 1], default=[0, 1], format_func=lambda x: "Sin diabetes" if x==0 else "Diabético")
        sexo_sel = st.multiselect("Sexo", options=[0, 1], default=[0, 1], format_func=lambda x: "Femenino" if x==0 else "Masculino")
        age_range = st.slider("Grupo de edad", int(df_raw["Age"].min()), int(df_raw["Age"].max()), (int(df_raw["Age"].min()), int(df_raw["Age"].max())))

    df = df_raw.copy()
    df = df[df["Diabetes_binary"].isin(clase_sel)]
    df = df[df["Sex"].isin(sexo_sel)]
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

    st.markdown(f'<div class="main-header"><h1>🩺 Predicción de Diabetes</h1><p>Dataset: BRFSS 2015 · Registros: {len(df):,}</p></div>', unsafe_allow_html=True)
    
    k1, k2, k3, k4 = st.columns(4)
    tasa = (df["Diabetes_binary"] == 1).mean() * 100
    for col, label, val in [(k1, "Registros", f"{len(df):,}"), (k2, "Tasa Diabetes", f"{tasa:.1f}%"), (k3, "IMC Promedio", f"{df['BMI'].mean():.1f}"), (k4, "Edad Prom.", f"{df['Age'].mean():.1f}")]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Descriptivas", "📈 Relaciones", "🧪 Pruebas"])
    
    with tab1:
        st.dataframe(df.describe().T.round(2), use_container_width=True)
        fig_pie = px.pie(df, names="Diabetes_binary", title="Distribución de Clases", color_discrete_sequence=["#4C9BE8", "#E86B4C"])
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        fig_bmi = px.histogram(df, x="BMI", color="Diabetes_binary", barmode="overlay", title="Distribución IMC por Clase")
        st.plotly_chart(fig_bmi, use_container_width=True)

    with tab3:
        resultados = []
        for var in VARS_NOMINALES:
            tabla = pd.crosstab(df[var], df["Diabetes_binary"])
            chi2, p, _, _ = chi2_contingency(tabla)
            resultados.append({"Variable": var, "p-valor": round(p, 6), "Significativa": "✅" if p < 0.05 else "❌"})
        st.table(pd.DataFrame(resultados))

# ════════════════════════════════════════════════════════════
# LÓGICA: REGRESIÓN AIRBNB
# ════════════════════════════════════════════════════════════
else:
    # Buscar path de Airbnb
    possible_paths = [CSV_NAME_AIRBNB, os.path.join("data", CSV_NAME_AIRBNB), os.path.join("..", "data", CSV_NAME_AIRBNB), os.path.join(os.path.dirname(__file__), "..", "data", CSV_NAME_AIRBNB)]
    DATA_PATH = next((p for p in possible_paths if os.path.exists(p)), None)

    if DATA_PATH is None:
        st.error(f"No se encontró `{CSV_NAME_AIRBNB}`.")
        st.stop()

    df_raw = cargar_datos_airbnb(DATA_PATH)

    with st.sidebar:
        st.subheader("🔎 Filtros")
        room_types = st.multiselect("Tipo Habitación", sorted(df_raw["room_type"].unique()), default=sorted(df_raw["room_type"].unique()))
        price_max = int(df_raw["price"].quantile(0.99)) if df_raw["price"].notna().any() else 10000
        price_range = st.slider("Precio (MXN)", 0, price_max, (0, price_max))

    df = df_raw.copy()
    df = df[df["room_type"].isin(room_types)]
    df = df[(df["price"].isna() | ((df["price"] >= price_range[0]) & (df["price"] <= price_range[1])))]

    st.markdown(f'<div class="main-header"><h1>🏠 Airbnb Listings — CDMX</h1><p>Análisis de Regresión · Registros: {len(df):,}</p></div>', unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    precio_med = df["price"].median()
    for col, label, val in [(k1, "Total Listings", f"{len(df):,}"), (k2, "Precio Mediano", f"${precio_med:,.0f}"), (k3, "Anfitriones", f"{df['host_id'].nunique():,}"), (k4, "Disponibilidad", f"{df['availability_365'].mean():.0f} d")]:
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Estadísticas", "📈 Gráficos", "🔬 Correlaciones"])

    with tab1:
        st.markdown('<div class="section-title">Resumen Numérico</div>', unsafe_allow_html=True)
        st.dataframe(df.describe().T.round(2), use_container_width=True)

    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(df[df["price"] < df["price"].quantile(0.95)], x="price", title="Distribución Precios", color_discrete_sequence=[AIRBNB_RED])
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_b:
            fig_map = px.scatter_mapbox(df.sample(min(2000, len(df))), lat="latitude", lon="longitude", color="price", size="price", zoom=10, mapbox_style="carto-positron")
            fig_map.update_layout(height=400)
            st.plotly_chart(fig_map, use_container_width=True)

    with tab3:
        vars_corr = ["price", "minimum_nights", "number_of_reviews", "availability_365"]
        corr_mat = df[vars_corr].corr()
        fig_heat = px.imshow(corr_mat, text_auto=True, color_continuous_scale="RdBu_r", title="Matriz de Correlación")
        st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.caption("Laboratorio Analítica de Datos · UdeA")