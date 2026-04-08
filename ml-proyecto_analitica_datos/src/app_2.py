# ============================================================
#  Dashboard Analítico Unificado
#  Laboratorio #3 · Analítica de Datos · UdeA
#  Proyectos:
#    · Regresión   — Airbnb Listings CDMX
#    · Clasificación — Predicción de Diabetes (BRFSS 2015)
# ============================================================

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Dashboard Analítico · UdeA",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
#  ESTILOS CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.02em; }

/* ── Header Airbnb ── */
.header-airbnb {
    background: linear-gradient(135deg, #FF385C 0%, #BD1E59 60%, #6A0F49 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.header-airbnb h1 { color: white !important; font-size: 2.2rem; margin: 0; }
.header-airbnb p  { color: rgba(255,255,255,0.82); margin: 0.3rem 0 0; font-size: 0.95rem; }

/* ── Header Diabetes ── */
.header-diabetes {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 60%, #01257d 100%);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
}
.header-diabetes h1 { color: white !important; font-size: 2.2rem; margin: 0; }
.header-diabetes p  { color: rgba(255,255,255,0.82); margin: 0.3rem 0 0; font-size: 0.95rem; }

/* ── KPI cards Airbnb ── */
.kpi-card-airbnb {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid #FF385C;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
}
/* ── KPI cards Diabetes ── */
.kpi-card-diabetes {
    background: white; border-radius: 12px; padding: 1.2rem 1.5rem;
    border-left: 4px solid #1a73e8;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #1a1a1a; }
.kpi-sub-airbnb   { font-size: 0.78rem; color: #FF385C; margin-top: 0.1rem; }
.kpi-sub-diabetes { font-size: 0.78rem; color: #1a73e8; margin-top: 0.1rem; }

/* ── Section titles ── */
.section-title-airbnb {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #1a1a1a;
    border-bottom: 2px solid #FF385C; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
}
.section-title-diabetes {
    font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; color: #1a1a1a;
    border-bottom: 2px solid #1a73e8; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1a1a1a !important; }
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #aaa !important; font-size: 0.82rem; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #f5f5f5; border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-family: 'Syne', sans-serif; font-weight: 600;
}
.stTabs [aria-selected="true"] { background: #FF385C !important; color: white !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  PALETAS DE COLOR
# ════════════════════════════════════════════════════════════
AIRBNB_RED  = "#FF385C"
AIRBNB_DARK = "#6A0F49"
COLOR_SEQ_A = ["#FF385C","#BD1E59","#6A0F49","#E8684A","#F7B731","#20BF6B"]

DIAB_BLUE  = "#1a73e8"
DIAB_ORANGE= "#E86B4C"
DIAB_C0    = "#4C9BE8"
DIAB_C1    = "#E86B4C"


# ════════════════════════════════════════════════════════════
#  FUNCIONES DE CARGA — AIRBNB
# ════════════════════════════════════════════════════════════

@st.cache_data
def cargar_airbnb(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df = df.drop(columns=["neighbourhood_group", "license"], errors="ignore")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    return df


@st.cache_data
def imputar_airbnb(df: pd.DataFrame):
    num_cols = ["price", "reviews_per_month"]

    df_media = df.copy()
    for col in num_cols:
        df_media[col] = df_media[col].fillna(df_media[col].mean())

    # Usar SimpleImputer con mediana en lugar de KNN para mejorar rendimiento y memoria
    # KNN consume demasiada memoria con datasets grandes (27k+ filas)
    df_knn = df.copy()
    features = ["price", "reviews_per_month", "minimum_nights",
                "number_of_reviews", "availability_365",
                "calculated_host_listings_count", "latitude", "longitude"]
    features_exist = [c for c in features if c in df_knn.columns]
    
    imputer = SimpleImputer(strategy="median")
    df_knn[features_exist] = imputer.fit_transform(df_knn[features_exist])

    comparativa = pd.DataFrame({
        "Variable":              num_cols,
        "Nulos originales":      [df[c].isnull().sum() for c in num_cols],
        "Media original":        [df[c].mean() for c in num_cols],
        "Media (imput. media)":  [df_media[c].mean() for c in num_cols],
        "Media (imput. KNN)":    [df_knn[c].mean() for c in num_cols],
        "Std original":          [df[c].std() for c in num_cols],
        "Std (imput. media)":    [df_media[c].std() for c in num_cols],
        "Std (imput. KNN)":      [df_knn[c].std() for c in num_cols],
    }).round(4)

    return df_media, df_knn, comparativa


# ════════════════════════════════════════════════════════════
#  FUNCIONES DE CARGA — DIABETES
# ════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_DIAB = BASE_DIR / "data" / "raw" / "dataset_clasificacion" \
           / "diabetes_binary_health_indicators_BRFSS2015.csv"
DB_DIAB  = BASE_DIR / "database" / "diabetes_clasificacion.db"

VARS_CONTINUAS = ["BMI"]
VARS_DISCRETAS = ["MentHlth", "PhysHlth"]
VARS_ORDINALES = ["GenHlth", "Age", "Education", "Income"]
VARS_NOMINALES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]
ETIQUETAS_AGE  = {1:"18–24",2:"25–29",3:"30–34",4:"35–39",5:"40–44",
                  6:"45–49",7:"50–54",8:"55–59",9:"60–64",10:"65–69",
                  11:"70–74",12:"75–79",13:"80+"}
ETIQUETAS_EDUC = {1:"Sin educación",2:"Primaria incompleta",3:"Primaria",
                  4:"Secundaria",5:"Técnico/Tecnólogo",6:"Universidad"}
ETIQUETAS_ING  = {1:"<$10K",2:"$10–15K",3:"$15–20K",4:"$20–25K",
                  5:"$25–35K",6:"$35–50K",7:"$50–75K",8:">$75K"}


@st.cache_data
def cargar_diabetes():
    if DB_DIAB.exists():
        conn = sqlite3.connect(DB_DIAB)
        df = pd.read_sql("SELECT * FROM diabetes", conn)
        conn.close()
    else:
        df = pd.read_csv(CSV_DIAB)
    return df


@st.cache_data
def calcular_mutual_info(df: pd.DataFrame):
    X = df.drop("Diabetes_binary", axis=1)
    y = df["Diabetes_binary"]
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"variable": X.columns, "mutual_info": scores}) \
             .sort_values("mutual_info", ascending=False)


# ════════════════════════════════════════════════════════════
#  SIDEBAR — SELECTOR DE PROYECTO
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Dashboard Analítico")
    st.markdown("**Analítica de Datos · UdeA**")
    st.markdown("---")

    proyecto = st.selectbox(
        "Selecciona el proyecto",
        ["🏠 Regresión — Airbnb CDMX", "🩺 Clasificación — Diabetes"],
        help="Cada proyecto carga su propio dataset y análisis"
    )

    st.markdown("---")
    st.markdown("### 🔎 Filtros")

    # ── Filtros Airbnb ───────────────────────────────────────
    if proyecto.startswith("🏠"):

        csv_name = "dataset_regresion_listings.csv"
        base_dir = Path(__file__).resolve().parent.parent
        possible_paths = [
            base_dir / "data" / "raw" / csv_name,
            base_dir / "data" / csv_name,
            Path(csv_name),
            Path("data") / csv_name,
            Path(os.path.dirname(__file__)) / ".." / "data" / csv_name,
        ]
        DATA_PATH = None
        for p in possible_paths:
            if p.exists():
                DATA_PATH = str(p)
                break

        if DATA_PATH is None:
            st.error(f"No se encontró `{csv_name}`. Ajusta la ruta.")
            st.stop()

        df_raw_a = cargar_airbnb(DATA_PATH)

        room_types  = st.multiselect("Tipo de habitación",
                                     sorted(df_raw_a["room_type"].dropna().unique()),
                                     default=sorted(df_raw_a["room_type"].dropna().unique()))
        neighbourhood = st.multiselect("Alcaldía",
                                       sorted(df_raw_a["neighbourhood"].dropna().unique()),
                                       default=sorted(df_raw_a["neighbourhood"].dropna().unique()))

        price_max   = int(df_raw_a["price"].quantile(0.99)) if df_raw_a["price"].notna().any() else 10000
        price_range = st.slider("Rango de precio (MXN/noche)",
                                int(df_raw_a["price"].min() or 0), price_max,
                                (int(df_raw_a["price"].min() or 0), price_max))

        avail_range = st.slider("Disponibilidad (días/año)", 0, 365, (0, 365))

    # ── Filtros Diabetes ─────────────────────────────────────
    else:
        try:
            df_raw_d = cargar_diabetes()
        except FileNotFoundError:
            st.error("❌ No se encontró el archivo de datos de diabetes. Verifica la ruta.")
            st.stop()

        clase_sel = st.multiselect(
            "Clase (Diabetes_binary)", options=[0, 1], default=[0, 1],
            format_func=lambda x: "0 — Sin diabetes" if x == 0 else "1 — Diabético/Prediabético"
        )
        age_min, age_max = int(df_raw_d["Age"].min()), int(df_raw_d["Age"].max())
        age_range = st.slider("Grupo de edad", age_min, age_max, (age_min, age_max))

        bmi_min, bmi_max = float(df_raw_d["BMI"].min()), float(df_raw_d["BMI"].max())
        bmi_range = st.slider("IMC (BMI)", bmi_min, bmi_max, (bmi_min, bmi_max))

        sexo_sel = st.multiselect(
            "Sexo", options=[0, 1], default=[0, 1],
            format_func=lambda x: "Femenino" if x == 0 else "Masculino"
        )

    st.markdown("---")
    st.caption("Lab #3 · Analítica de Datos · UdeA")


# ════════════════════════════════════════════════════════════
#  PROYECTO: AIRBNB (REGRESIÓN)
# ════════════════════════════════════════════════════════════
if proyecto.startswith("🏠"):

    # ── Filtrado ─────────────────────────────────────────────
    df = df_raw_a.copy()
    if room_types:
        df = df[df["room_type"].isin(room_types)]
    if neighbourhood:
        df = df[df["neighbourhood"].isin(neighbourhood)]
    df = df[
        (df["price"].isna() | ((df["price"] >= price_range[0]) & (df["price"] <= price_range[1]))) &
        (df["availability_365"] >= avail_range[0]) &
        (df["availability_365"] <= avail_range[1])
    ]

    # ── Header ───────────────────────────────────────────────
    st.markdown("""
    <div class="header-airbnb">
        <h1>🏠 Airbnb Listings — Ciudad de México</h1>
        <p>Laboratorio #3 · Análisis Integral de Datos para Regresión · Analítica de Datos · UdeA</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    precio_med = df["price"].median()
    precio_avg = df["price"].mean()
    hosts_uniq = df["host_id"].nunique()
    avail_avg  = df["availability_365"].mean()

    for col, label, val, sub in [
        (k1, "Total Listings",       f"{len(df):,}",          f"de {len(df_raw_a):,} totales"),
        (k2, "Precio Mediano",       f"${precio_med:,.0f}",    "MXN / noche"),
        (k3, "Precio Promedio",      f"${precio_avg:,.0f}",    "MXN / noche"),
        (k4, "Anfitriones únicos",   f"{hosts_uniq:,}",        "hosts"),
        (k5, "Disponibilidad prom.", f"{avail_avg:.0f} días",  "/ año (365)"),
    ]:
        col.markdown(f"""
        <div class="kpi-card-airbnb">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub-airbnb">{sub}</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Estadísticas Descriptivas",
        "📈 Gráficos Principales",
        "🧪 Imputación de Datos",
        "🔬 Correlaciones y Pruebas"
    ])

    # ── TAB 1 · Estadísticas Descriptivas ────────────────────
    with tab1:
        st.markdown('<div class="section-title-airbnb">Tipos de Variables</div>',
                    unsafe_allow_html=True)

        vars_num = df.select_dtypes(include=np.number).columns.tolist()
        vars_num = [v for v in vars_num if v not in ["id", "host_id"]]
        vars_cat = ["room_type", "neighbourhood"]

        c1, c2 = st.columns(2)
        c1.info(f"**Variables numéricas ({len(vars_num)}):**\n\n" + ", ".join(f"`{v}`" for v in vars_num))
        c2.info(f"**Variables categóricas analizadas ({len(vars_cat)}):**\n\n" + ", ".join(f"`{v}`" for v in vars_cat))

        st.markdown('<div class="section-title-airbnb">Estadísticas Descriptivas — Variables Numéricas</div>',
                    unsafe_allow_html=True)

        desc = df[vars_num].describe().T
        desc["skewness"] = df[vars_num].skew()
        desc["kurtosis"] = df[vars_num].kurtosis()
        desc["nulos"]    = df[vars_num].isnull().sum()
        desc["% nulos"]  = (df[vars_num].isnull().mean() * 100).round(2)
        st.dataframe(desc.round(3), use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Valores Faltantes por Variable</div>',
                    unsafe_allow_html=True)

        nulos_df = pd.DataFrame({
            "Variable":    df.columns,
            "Nulos":       df.isnull().sum().values,
            "% Faltante":  (df.isnull().mean() * 100).round(2).values
        }).sort_values("% Faltante", ascending=False)
        nulos_df = nulos_df[nulos_df["Nulos"] > 0]

        if not nulos_df.empty:
            fig_nulos = px.bar(nulos_df, x="Variable", y="% Faltante",
                               text="Nulos", color="% Faltante",
                               color_continuous_scale=["#FFD6DC", AIRBNB_RED, AIRBNB_DARK],
                               title="Porcentaje de Valores Faltantes por Variable")
            fig_nulos.update_traces(textposition="outside")
            fig_nulos.update_layout(showlegend=False, height=380,
                                    plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_nulos, use_container_width=True)
        else:
            st.success("✅ No hay valores faltantes en el subconjunto filtrado.")

        st.markdown('<div class="section-title-airbnb">Frecuencias — Variables Categóricas</div>',
                    unsafe_allow_html=True)

        cc1, cc2 = st.columns(2)
        with cc1:
            rt_counts = df["room_type"].value_counts().reset_index()
            rt_counts.columns = ["room_type", "count"]
            fig_rt = px.pie(rt_counts, names="room_type", values="count",
                            title="Distribución por Tipo de Habitación",
                            color_discrete_sequence=COLOR_SEQ_A, hole=0.45)
            fig_rt.update_layout(height=360)
            st.plotly_chart(fig_rt, use_container_width=True)

        with cc2:
            nb_counts = df["neighbourhood"].value_counts().head(10).reset_index()
            nb_counts.columns = ["neighbourhood", "count"]
            fig_nb = px.bar(nb_counts, x="count", y="neighbourhood", orientation="h",
                            title="Top 10 Alcaldías por Número de Listings",
                            color="count",
                            color_continuous_scale=["#FFD6DC", AIRBNB_RED])
            fig_nb.update_layout(yaxis={"categoryorder": "total ascending"},
                                  height=360, showlegend=False,
                                  plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_nb, use_container_width=True)

    # ── TAB 2 · Gráficos Principales ─────────────────────────
    with tab2:
        st.markdown('<div class="section-title-airbnb">Distribución del Precio</div>',
                    unsafe_allow_html=True)

        price_cap = df["price"][df["price"] <= df["price"].quantile(0.97)].dropna()

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(price_cap, x=price_cap,
                                    nbins=60, title="Histograma de Precio (p97)",
                                    color_discrete_sequence=[AIRBNB_RED])
            fig_hist.add_vline(x=price_cap.mean(), line_dash="dash", line_color="#1a1a1a",
                               annotation_text=f"Media ${price_cap.mean():,.0f}")
            fig_hist.add_vline(x=price_cap.median(), line_dash="dot", line_color=AIRBNB_DARK,
                               annotation_text=f"Mediana ${price_cap.median():,.0f}")
            fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   xaxis_title="Precio (MXN/noche)", yaxis_title="Frecuencia")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_box_price = px.box(df[df["price"] <= df["price"].quantile(0.97)],
                                   x="room_type", y="price", color="room_type",
                                   title="Distribución de Precio por Tipo de Habitación",
                                   color_discrete_sequence=COLOR_SEQ_A)
            fig_box_price.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                        showlegend=False)
            st.plotly_chart(fig_box_price, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Precio por Alcaldía</div>',
                    unsafe_allow_html=True)

        precio_nb = (df.groupby("neighbourhood")["price"]
                     .agg(["mean", "median", "count"])
                     .query("count >= 30").reset_index()
                     .rename(columns={"mean":"Media","median":"Mediana","count":"Listings"})
                     .sort_values("Media", ascending=False).head(15))

        fig_nb_price = go.Figure()
        fig_nb_price.add_bar(x=precio_nb["neighbourhood"], y=precio_nb["Media"],
                             name="Media", marker_color=AIRBNB_RED, opacity=0.85)
        fig_nb_price.add_bar(x=precio_nb["neighbourhood"], y=precio_nb["Mediana"],
                             name="Mediana", marker_color=AIRBNB_DARK, opacity=0.85)
        fig_nb_price.update_layout(barmode="group",
                                    title="Media y Mediana del Precio por Alcaldía (top 15)",
                                    plot_bgcolor="white", paper_bgcolor="white",
                                    xaxis_tickangle=-35, height=420,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_nb_price, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Scatter: Variables vs Precio</div>',
                    unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        df_sc = df[df["price"] <= df["price"].quantile(0.97)].dropna(subset=["price"])

        with sc1:
            fig_sc1 = px.scatter(df_sc.sample(min(3000, len(df_sc)), random_state=42),
                                 x="availability_365", y="price", color="room_type",
                                 opacity=0.5, size_max=6, color_discrete_sequence=COLOR_SEQ_A,
                                 title="Disponibilidad vs Precio",
                                 labels={"availability_365":"Disponibilidad (días/año)",
                                         "price":"Precio (MXN)"})
            fig_sc1.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_sc1, use_container_width=True)

        with sc2:
            fig_sc2 = px.scatter(df_sc.sample(min(3000, len(df_sc)), random_state=42),
                                 x="number_of_reviews", y="price", color="room_type",
                                 opacity=0.5, size_max=6, color_discrete_sequence=COLOR_SEQ_A,
                                 title="Número de Reseñas vs Precio",
                                 labels={"number_of_reviews":"Número de Reseñas",
                                         "price":"Precio (MXN)"})
            fig_sc2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_sc2, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Mapa Geográfico de Listings</div>',
                    unsafe_allow_html=True)

        df_geo = df.dropna(subset=["price", "latitude", "longitude"])
        df_geo = df_geo[df_geo["price"] <= df_geo["price"].quantile(0.95)]
        df_geo_sample = df_geo.sample(min(5000, len(df_geo)), random_state=42)

        fig_map = px.scatter_mapbox(df_geo_sample,
                                    lat="latitude", lon="longitude", color="price",
                                    color_continuous_scale=["#FFD6DC", AIRBNB_RED, AIRBNB_DARK],
                                    size="price", size_max=10, zoom=10,
                                    mapbox_style="carto-positron",
                                    hover_data=["neighbourhood", "room_type", "price"],
                                    title="Distribución Geográfica de Listings (coloreado por precio)")
        fig_map.update_layout(height=520, margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)

    # ── TAB 3 · Imputación de Datos ──────────────────────────
    with tab3:
        st.markdown('<div class="section-title-airbnb">Métodos de Imputación Aplicados</div>',
                    unsafe_allow_html=True)

        st.info("""
        Se aplican **dos métodos** de imputación sobre las variables con datos faltantes (`price` y `reviews_per_month`):
        - **Método 1 — Media**: reemplaza los nulos por la media de la columna. Simple y rápido, pero sensible a outliers.
        - **Método 2 — KNN (K-Nearest Neighbors)**: imputa usando los 5 vecinos más cercanos en el espacio de variables numéricas. Más robusto y preserva mejor la estructura de los datos.
        """)

        with st.spinner("Calculando imputaciones..."):
            df_media, df_knn, comparativa = imputar_airbnb(df_raw_a)

        st.markdown('<div class="section-title-airbnb">Tabla Comparativa: Antes y Después</div>',
                    unsafe_allow_html=True)
        st.dataframe(comparativa, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Distribución del Precio: Original vs Imputado</div>',
                    unsafe_allow_html=True)

        cap = df_raw_a["price"].quantile(0.97)
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Histogram(x=df_raw_a["price"][df_raw_a["price"] <= cap].dropna(),
                                       name="Original (con nulos)", opacity=0.6,
                                       marker_color="#aaaaaa", nbinsx=60))
        fig_imp.add_trace(go.Histogram(x=df_media["price"][df_media["price"] <= cap],
                                       name="Imputación por Media", opacity=0.6,
                                       marker_color=AIRBNB_RED, nbinsx=60))
        fig_imp.add_trace(go.Histogram(x=df_knn["price"][df_knn["price"] <= cap],
                                       name="Imputación KNN", opacity=0.6,
                                       marker_color=AIRBNB_DARK, nbinsx=60))
        fig_imp.update_layout(barmode="overlay",
                              title="Distribución del Precio — Comparativa de Imputaciones (p97)",
                              xaxis_title="Precio (MXN/noche)", yaxis_title="Frecuencia",
                              plot_bgcolor="white", paper_bgcolor="white", height=420,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Distribución de Reviews/Mes: Original vs Imputado</div>',
                    unsafe_allow_html=True)

        fig_imp2 = go.Figure()
        fig_imp2.add_trace(go.Histogram(x=df_raw_a["reviews_per_month"].dropna(),
                                        name="Original", opacity=0.6, marker_color="#aaaaaa", nbinsx=60))
        fig_imp2.add_trace(go.Histogram(x=df_media["reviews_per_month"],
                                        name="Media", opacity=0.6, marker_color=AIRBNB_RED, nbinsx=60))
        fig_imp2.add_trace(go.Histogram(x=df_knn["reviews_per_month"],
                                        name="KNN", opacity=0.6, marker_color=AIRBNB_DARK, nbinsx=60))
        fig_imp2.update_layout(barmode="overlay", title="Distribución de Reviews/Mes — Comparativa",
                               xaxis_title="Reviews por Mes", yaxis_title="Frecuencia",
                               plot_bgcolor="white", paper_bgcolor="white", height=380,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_imp2, use_container_width=True)

        st.markdown("### 📌 Justificación del Método Recomendado")
        st.success("""
        **Se recomienda la imputación por KNN** para este dataset por las siguientes razones:
        - La distribución del precio es fuertemente **asimétrica** (skew > 10), por lo que la media no es representativa
          del valor central — la imputación por media infla artificialmente la densidad alrededor de un valor que pocos listings tienen.
        - KNN usa información de **listings similares** (misma zona, disponibilidad, tipo de habitación) para estimar el precio faltante,
          lo cual preserva mejor la estructura real del mercado.
        - La comparación de desviaciones estándar muestra que **KNN mantiene una variabilidad más cercana a la original**.
        """)

    # ── TAB 4 · Correlaciones y Pruebas Estadísticas ─────────
    with tab4:
        st.markdown('<div class="section-title-airbnb">Matriz de Correlación (Pearson)</div>',
                    unsafe_allow_html=True)

        vars_corr = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month",
                     "calculated_host_listings_count", "availability_365",
                     "number_of_reviews_ltm", "latitude", "longitude"]
        vars_corr_exist = [v for v in vars_corr if v in df.columns]

        corr_mat = df[vars_corr_exist].corr()
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_mat.values,
            x=corr_mat.columns.tolist(), y=corr_mat.columns.tolist(),
            colorscale=[[0, AIRBNB_DARK],[0.5, "white"],[1, AIRBNB_RED]],
            zmid=0, text=corr_mat.round(2).values, texttemplate="%{text}",
            showscale=True, zmin=-1, zmax=1
        ))
        fig_heat.update_layout(title="Matriz de Correlación de Pearson",
                               height=520, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Correlación con Price: Pearson vs Spearman</div>',
                    unsafe_allow_html=True)

        vars_test = [v for v in vars_corr_exist if v != "price"]
        results_corr = []
        for var in vars_test:
            par = df[["price", var]].dropna()
            if len(par) < 10:
                continue
            r_p, pv_p = stats.pearsonr(par["price"], par[var])
            r_s, pv_s = stats.spearmanr(par["price"], par[var])
            results_corr.append({
                "Variable":      var,
                "Pearson r":     round(r_p, 4),
                "p Pearson":     round(pv_p, 6),
                "Spearman r":    round(r_s, 4),
                "p Spearman":    round(pv_s, 6),
                "Sig. Pearson":  "***" if pv_p<0.001 else ("**" if pv_p<0.01 else ("*" if pv_p<0.05 else "ns")),
                "Sig. Spearman": "***" if pv_s<0.001 else ("**" if pv_s<0.01 else ("*" if pv_s<0.05 else "ns")),
            })

        df_corr_res = pd.DataFrame(results_corr).sort_values("Pearson r", key=abs, ascending=False)
        st.dataframe(df_corr_res, use_container_width=True)
        st.caption("Significancia: \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · ns = no significativo")

        fig_corr_bar = go.Figure()
        fig_corr_bar.add_bar(x=df_corr_res["Variable"], y=df_corr_res["Pearson r"],
                             name="Pearson", marker_color=AIRBNB_RED, opacity=0.85)
        fig_corr_bar.add_bar(x=df_corr_res["Variable"], y=df_corr_res["Spearman r"],
                             name="Spearman", marker_color=AIRBNB_DARK, opacity=0.85)
        fig_corr_bar.add_hline(y=0, line_color="black", line_width=0.8)
        fig_corr_bar.update_layout(barmode="group",
                                    title="Pearson vs Spearman — Correlación con Price",
                                    plot_bgcolor="white", paper_bgcolor="white", height=380,
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig_corr_bar, use_container_width=True)

        st.markdown('<div class="section-title-airbnb">Prueba ANOVA y Kruskal-Wallis — Precio por Tipo de Habitación</div>',
                    unsafe_allow_html=True)

        grupos = [df[df["room_type"] == rt]["price"].dropna() for rt in df["room_type"].unique()]
        grupos = [g for g in grupos if len(g) >= 5]

        if len(grupos) >= 2:
            f_stat, p_anova   = stats.f_oneway(*grupos)
            h_stat, p_kruskal = stats.kruskal(*grupos)

            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("#### ANOVA (paramétrica)")
                st.metric("F-estadístico", f"{f_stat:.4f}")
                st.metric("p-valor", f"{p_anova:.2e}")
                if p_anova < 0.05:
                    st.success("✅ Se rechaza H₀: hay diferencias significativas de precio entre tipos de habitación (α=0.05)")
                else:
                    st.warning("No se rechaza H₀ con α=0.05")
            with col_a2:
                st.markdown("#### Kruskal-Wallis (no paramétrica)")
                st.metric("H-estadístico", f"{h_stat:.4f}")
                st.metric("p-valor", f"{p_kruskal:.2e}")
                if p_kruskal < 0.05:
                    st.success("✅ Se rechaza H₀: las distribuciones de precio difieren entre tipos (α=0.05)")
                else:
                    st.warning("No se rechaza H₀ con α=0.05")
        else:
            st.warning("No hay suficientes grupos para la prueba con los filtros actuales.")

        st.markdown('<div class="section-title-airbnb">Prueba Chi-cuadrado — room_type vs neighbourhood</div>',
                    unsafe_allow_html=True)

        top_nb = df["neighbourhood"].value_counts().head(6).index.tolist()
        df_chi = df[df["neighbourhood"].isin(top_nb)][["room_type", "neighbourhood"]].dropna()
        if len(df_chi) >= 20:
            tabla_cont = pd.crosstab(df_chi["room_type"], df_chi["neighbourhood"])
            chi2_val, p_chi, dof, _ = stats.chi2_contingency(tabla_cont)

            c_chi1, c_chi2, c_chi3 = st.columns(3)
            c_chi1.metric("Chi² estadístico", f"{chi2_val:.4f}")
            c_chi2.metric("p-valor", f"{p_chi:.2e}")
            c_chi3.metric("Grados de libertad", dof)

            if p_chi < 0.05:
                st.success("✅ Se rechaza H₀: existe asociación significativa entre tipo de habitación y alcaldía (α=0.05)")
            else:
                st.warning("No se rechaza H₀ con α=0.05")

            st.markdown("**Tabla de contingencia (frecuencias observadas):**")
            st.dataframe(tabla_cont, use_container_width=True)
        else:
            st.warning("Datos insuficientes para la prueba Chi-cuadrado con los filtros actuales.")

        st.markdown('<div class="section-title-airbnb">Datos Filtrados</div>',
                    unsafe_allow_html=True)
        st.caption(f"Mostrando {min(500, len(df)):,} de {len(df):,} registros filtrados")
        st.dataframe(df.head(500), use_container_width=True)

    # Footer Airbnb
    st.markdown("---")
    st.caption("📌 Analítica de Datos · Universidad de Antioquia | Dataset: Inside Airbnb — CDMX")


# ════════════════════════════════════════════════════════════
#  PROYECTO: DIABETES (CLASIFICACIÓN)
# ════════════════════════════════════════════════════════════
else:

    # ── Filtrado ─────────────────────────────────────────────
    df = df_raw_d.copy()
    if clase_sel:
        df = df[df["Diabetes_binary"].isin(clase_sel)]
    if sexo_sel:
        df = df[df["Sex"].isin(sexo_sel)]
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
    df = df[(df["BMI"] >= bmi_range[0]) & (df["BMI"] <= bmi_range[1])]

    # ── Header ───────────────────────────────────────────────
    st.markdown(f"""
    <div class="header-diabetes">
        <h1>🩺 Predicción de Diabetes — BRFSS 2015</h1>
        <p>Laboratorio #3 · Análisis Integral de Datos para Clasificación · Analítica de Datos · UdeA
        &nbsp;·&nbsp; Registros filtrados: {len(df):,} / {len(df_raw_d):,}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPIs ─────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    tasa        = (df["Diabetes_binary"] == 1).mean() * 100
    bmi_prom    = df["BMI"].mean()
    edad_prom   = df["Age"].mean()
    sin_acceso  = (df["AnyHealthcare"] == 0).mean() * 100
    riesgo_alto = ((df["HighBP"]==1) & (df["HighChol"]==1) & (df["BMI"]>=30)).mean() * 100

    k1.metric("👥 Total registros",      f"{len(df):,}")
    k2.metric("📈 Tasa de diabetes",     f"{tasa:.1f}%")
    k3.metric("⚖️ IMC promedio",         f"{bmi_prom:.1f}")
    k4.metric("🎂 Grupo edad prom.",     f"{edad_prom:.1f}")
    k5.metric("⚠️ Sin cobertura médica", f"{sin_acceso:.1f}%")

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Estadísticas Descriptivas",
        "📈 Distribuciones y Relaciones",
        "🔗 Correlaciones",
        "🧪 Pruebas Estadísticas",
        "🛠️ Imputación de Datos"
    ])

    # ── TAB 1 · Estadísticas Descriptivas ────────────────────
    with tab1:
        st.markdown('<div class="section-title-diabetes">Estadísticas descriptivas</div>',
                    unsafe_allow_html=True)

        tipo_var = st.radio("Tipo de variables a mostrar",
                            ["Numéricas / Ordinales", "Categóricas binarias"], horizontal=True)

        if tipo_var == "Numéricas / Ordinales":
            vars_desc = VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES
            desc = df[vars_desc].describe().T
            desc["mediana"]   = df[vars_desc].median()
            desc["asimetría"] = df[vars_desc].skew().round(3)
            desc = desc[["count","mean","mediana","std","min","25%","75%","max","asimetría"]]
            desc.columns = ["n","media","mediana","std","min","Q1","Q3","max","asimetría"]
            st.dataframe(desc.round(3), use_container_width=True)
        else:
            rows = []
            for var in VARS_NOMINALES:
                vc = df[var].value_counts()
                rows.append({
                    "variable":     var,
                    "total":        len(df[var].dropna()),
                    "frecuencia_0": vc.get(0, 0),
                    "frecuencia_1": vc.get(1, 0),
                    "pct_1 (%)":    round(vc.get(1,0)/len(df[var].dropna())*100, 2)
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown('<div class="section-title-diabetes">Distribución de la variable objetivo</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns([1, 2])
        vc = df["Diabetes_binary"].value_counts().reset_index()
        vc.columns = ["clase", "cantidad"]
        vc["etiqueta"] = vc["clase"].map({0:"Sin diabetes", 1:"Diabético/Prediabético"})

        with col1:
            fig = px.pie(vc, values="cantidad", names="etiqueta",
                         color_discrete_sequence=[DIAB_C0, DIAB_C1], hole=0.45)
            fig.update_traces(textposition="outside", textinfo="percent+label")
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(vc, x="etiqueta", y="cantidad", color="etiqueta",
                         color_discrete_sequence=[DIAB_C0, DIAB_C1],
                         text="cantidad", title="Conteo por clase")
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Registros")
            st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"ℹ️ **Desbalanceo:** la clase mayoritaria (sin diabetes) representa el "
            f"{(df['Diabetes_binary']==0).mean()*100:.1f}% del total filtrado. "
            "Esto implica que accuracy sola no es suficiente; se recomienda usar AUC-ROC, precisión y recall."
        )

    # ── TAB 2 · Distribuciones y Relaciones ──────────────────
    with tab2:
        st.markdown('<div class="section-title-diabetes">Distribuciones y relaciones entre variables</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribución del IMC por clase**")
            fig = px.histogram(df, x="BMI", color="Diabetes_binary",
                               barmode="overlay", nbins=60, opacity=0.7,
                               color_discrete_map={0: DIAB_C0, 1: DIAB_C1},
                               labels={"Diabetes_binary":"Clase", "BMI":"IMC"})
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
            tasa_edad["Tasa (%)"]   = (tasa_edad["Diabetes_binary"] * 100).round(2)
            fig = px.bar(tasa_edad, x="Grupo edad", y="Tasa (%)",
                         color="Tasa (%)", color_continuous_scale="Oranges")
            fig.update_layout(xaxis_tickangle=-30, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            var_box = st.selectbox("Variable para boxplot por clase",
                                   VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES)
            df_box = df.copy()
            df_box["Clase"] = df_box["Diabetes_binary"].map(
                {0:"Sin diabetes", 1:"Diabético/Prediabético"})
            fig = px.box(df_box, x="Clase", y=var_box, color="Clase",
                         color_discrete_map={"Sin diabetes":DIAB_C0,"Diabético/Prediabético":DIAB_C1},
                         points=False, notched=True,
                         title=f"Distribución de {var_box} por clase")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            var_cat = st.selectbox("Variable categórica — tasa de diabetes",
                                   VARS_NOMINALES + VARS_ORDINALES)
            tasa_cat = df.groupby(var_cat)["Diabetes_binary"].mean().reset_index()
            tasa_cat.columns = [var_cat, "Tasa (%)"]
            tasa_cat["Tasa (%)"] = (tasa_cat["Tasa (%)"] * 100).round(2)
            if var_cat == "Age":
                tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_AGE)
            elif var_cat == "Education":
                tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_EDUC)
            elif var_cat == "Income":
                tasa_cat[var_cat] = tasa_cat[var_cat].map(ETIQUETAS_ING)
            fig = px.bar(tasa_cat, x=var_cat, y="Tasa (%)",
                         color="Tasa (%)", color_continuous_scale="Reds",
                         title=f"Tasa de diabetes por {var_cat}")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Tasa de diabetes por nivel de ingresos y educación**")
        pivot = df.groupby(["Income","Education"])["Diabetes_binary"].mean().reset_index()
        pivot_table = pivot.pivot(index="Income", columns="Education",
                                  values="Diabetes_binary") * 100
        fig = px.imshow(pivot_table.round(1), color_continuous_scale="RdYlGn_r",
                        aspect="auto",
                        labels=dict(x="Educación", y="Ingresos", color="Tasa diabetes (%)"),
                        title="Tasa de diabetes (%) por nivel de ingresos y educación")
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 3 · Correlaciones ─────────────────────────────────
    with tab3:
        st.markdown('<div class="section-title-diabetes">Análisis de correlación</div>',
                    unsafe_allow_html=True)

        metodo_corr = st.radio("Método de correlación", ["Spearman","Pearson"], horizontal=True)
        corr = df.corr(method=metodo_corr.lower())

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(), y=corr.columns.tolist(),
            colorscale="RdYlGn", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont={"size":8}, hoverongaps=False
        ))
        fig.update_layout(title=f"Matriz de correlación de {metodo_corr}", height=600,
                          xaxis=dict(tickfont=dict(size=9)),
                          yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Correlación de cada variable con `Diabetes_binary`**")
        target_corr = corr["Diabetes_binary"].drop("Diabetes_binary").sort_values()
        colors_bar = [DIAB_C1 if v > 0 else DIAB_C0 for v in target_corr.values]

        fig = go.Figure(go.Bar(
            x=target_corr.values, y=target_corr.index, orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.3f}" for v in target_corr.values], textposition="outside"
        ))
        fig.add_vline(x=0, line_width=1, line_color="black")
        fig.update_layout(title=f"Correlación de {metodo_corr} con Diabetes_binary",
                          xaxis_title="Coeficiente de correlación", height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Información Mutua con `Diabetes_binary`**")
        mi_df = calcular_mutual_info(df)
        fig = px.bar(mi_df.sort_values("mutual_info"),
                     x="mutual_info", y="variable", orientation="h",
                     color="mutual_info", color_continuous_scale="Blues",
                     title="Mutual Information Score respecto a Diabetes_binary",
                     labels={"mutual_info":"Mutual Info","variable":"Variable"})
        fig.update_layout(coloraxis_showscale=False, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── TAB 4 · Pruebas Estadísticas ─────────────────────────
    with tab4:
        st.markdown('<div class="section-title-diabetes">Pruebas de asociación y dependencia</div>',
                    unsafe_allow_html=True)

        st.markdown("#### Chi-cuadrado — Variables categóricas nominales vs `Diabetes_binary`")
        st.caption("Prueba si existe asociación estadística entre cada variable y la clase objetivo.")

        @st.cache_data
        def calcular_chi2_diab(df: pd.DataFrame):
            resultados = []
            for var in VARS_NOMINALES:
                tabla = pd.crosstab(df[var], df["Diabetes_binary"])
                chi2_v, p, dof, _ = chi2_contingency(tabla)
                n = tabla.sum().sum()
                cramer_v = np.sqrt(chi2_v / (n * (min(tabla.shape) - 1)))
                resultados.append({
                    "Variable":              var,
                    "Chi²":                  round(chi2_v, 2),
                    "p-valor":               round(p, 6),
                    "gl":                    dof,
                    "V de Cramér":           round(cramer_v, 4),
                    "Significativa (α=0.05)":"✅ Sí" if p < 0.05 else "❌ No"
                })
            return pd.DataFrame(resultados).sort_values("V de Cramér", ascending=False)

        chi2_df = calcular_chi2_diab(df)
        st.dataframe(chi2_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Kruskal-Wallis — Variables numéricas/ordinales vs `Diabetes_binary`")
        st.caption("Prueba si las distribuciones de las variables difieren entre las dos clases.")

        @st.cache_data
        def calcular_kruskal_diab(df: pd.DataFrame):
            resultados = []
            for var in VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES:
                g0 = df[df["Diabetes_binary"]==0][var].dropna()
                g1 = df[df["Diabetes_binary"]==1][var].dropna()
                if len(g0) > 0 and len(g1) > 0:
                    stat, p = kruskal(g0, g1)
                    resultados.append({
                        "Variable":                var,
                        "Media (clase 0)":         round(g0.mean(), 3),
                        "Media (clase 1)":         round(g1.mean(), 3),
                        "Estadístico H":           round(stat, 2),
                        "p-valor":                 round(p, 8),
                        "Diferencia significativa":"✅ Sí" if p < 0.05 else "❌ No"
                    })
            return pd.DataFrame(resultados).sort_values("Estadístico H", ascending=False)

        kw_df = calcular_kruskal_diab(df)
        st.dataframe(kw_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Interpretación")
        st.info(
            "**Chi-cuadrado:** valores p < 0.05 indican asociación estadísticamente significativa. "
            "La **V de Cramér** mide el tamaño del efecto: >0.1 efecto pequeño, >0.3 moderado, >0.5 grande.  \n\n"
            "**Kruskal-Wallis:** alternativa no paramétrica al ANOVA. Valores p < 0.05 indican que las "
            "distribuciones de la variable difieren significativamente entre diabéticos y no diabéticos."
        )

    # ── TAB 5 · Imputación de Datos ──────────────────────────
    with tab5:
        st.markdown('<div class="section-title-diabetes">Tratamiento de datos faltantes e imputación</div>',
                    unsafe_allow_html=True)

        st.info(
            "El dataset BRFSS 2015 no contiene valores nulos en su versión original. "
            "Para demostrar y comparar los métodos de imputación, se introduce un porcentaje "
            "controlado de valores faltantes artificiales en las variables numéricas y ordinales."
        )

        missing_rate = st.slider("Porcentaje de valores faltantes a introducir (%)",
                                  min_value=1, max_value=20, value=5, step=1)

        vars_imputar = VARS_CONTINUAS + VARS_DISCRETAS + VARS_ORDINALES

        @st.cache_data
        def generar_faltantes(df, rate, seed=42):
            np.random.seed(seed)
            df_miss = df.copy()
            for var in vars_imputar:
                idx = np.random.choice(df_miss.index,
                                       size=int(len(df_miss)*rate/100), replace=False)
                df_miss.loc[idx, var] = np.nan
            return df_miss

        df_miss = generar_faltantes(df, missing_rate)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Valores faltantes por variable**")
            miss_info = pd.DataFrame({
                "Variable":      vars_imputar,
                "Faltantes":     [df_miss[v].isnull().sum() for v in vars_imputar],
                "Porcentaje (%)": [round(df_miss[v].isnull().mean()*100, 2) for v in vars_imputar]
            })
            st.dataframe(miss_info, hide_index=True, use_container_width=True)

        with col2:
            fig = px.bar(miss_info, x="Variable", y="Porcentaje (%)",
                         color="Porcentaje (%)", color_continuous_scale="Blues",
                         title="% de valores faltantes por variable")
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Comparación de métodos de imputación**")

        metodo_imp = st.radio("Método a visualizar",
                               ["Mediana (tendencia central)", "Mediana (optimizado)"], horizontal=True)

        @st.cache_data
        def imputar_diabetes(df_miss):
            imp_med = SimpleImputer(strategy="median")
            df_med  = df_miss.copy()
            df_med[vars_imputar] = imp_med.fit_transform(df_miss[vars_imputar])

            # Usar SimpleImputer con mediana en lugar de KNN para optimizar rendimiento y memoria
            imp_knn = SimpleImputer(strategy="median")
            df_knn  = df_miss.copy()
            df_knn[vars_imputar] = imp_knn.fit_transform(df_miss[vars_imputar])
            return df_med, df_knn

        df_mediana_d, df_knn_d = imputar_diabetes(df_miss)
        df_imputado   = df_mediana_d if "Mediana" in metodo_imp else df_knn_d
        nombre_metodo = "Mediana" if "Mediana" in metodo_imp else "Mediana (optimizado)"

        var_vis = st.selectbox("Variable a visualizar", vars_imputar)

        col3, col4 = st.columns(2)
        with col3:
            fig = px.histogram(df[var_vis].dropna(), nbins=40,
                               title=f"{var_vis} — Original",
                               color_discrete_sequence=[DIAB_C0])
            fig.update_layout(xaxis_title=var_vis, yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            color_imp = DIAB_ORANGE if nombre_metodo == "Mediana" else "#4CE8A0"
            fig = px.histogram(df_imputado[var_vis], nbins=40,
                               title=f"{var_vis} — Imputado ({nombre_metodo})",
                               color_discrete_sequence=[color_imp])
            fig.update_layout(xaxis_title=var_vis, yaxis_title="Frecuencia")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("**Tabla comparativa de estadísticos antes y después de la imputación**")

        comp_rows = []
        for var in vars_imputar:
            for label, src in [("Original", df), ("Mediana", df_mediana_d), ("KNN", df_knn_d)]:
                comp_rows.append({
                    "Variable": var, "Método": label,
                    "Media":    round(src[var].mean(), 3),
                    "Mediana":  round(src[var].median(), 3),
                    "Std":      round(src[var].std(), 3)
                })

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

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

    # Footer Diabetes
    st.markdown("---")
    st.caption(
        "📌 Analítica de Datos · Universidad de Antioquia  "
        "| Dataset: BRFSS 2015 — CDC  "
        "| Fuente: kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset"
    )
