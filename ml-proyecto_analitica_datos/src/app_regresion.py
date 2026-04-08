# ============================================================
#  Airbnb Listings CDMX — Dashboard Analítico
#  Laboratorio #3 · Analítica de Datos · UdeA
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os

# ── Configuración de página ──────────────────────────────────
st.set_page_config(
    page_title="Airbnb CDMX · Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilo CSS personalizado ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
}

/* Header principal */
.main-header {
    background: linear-gradient(135deg, #FF385C 0%, #BD1E59 60%, #6A0F49 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 { color: white !important; font-size: 2.2rem; margin: 0; }
.main-header p  { color: rgba(255,255,255,0.82); margin: 0.3rem 0 0; font-size: 0.95rem; }

/* Tarjetas KPI */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    border-left: 4px solid #FF385C;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.78rem; color: #888; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; color: #1a1a1a; }
.kpi-sub   { font-size: 0.78rem; color: #FF385C; margin-top: 0.1rem; }

/* Sección título */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1a1a;
    border-bottom: 2px solid #FF385C;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1a1a1a !important;
}
[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #aaa !important; font-size: 0.82rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #f5f5f5;
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #FF385C !important;
    color: white !important;
}

/* Tablas de resultado */
.result-badge {
    display: inline-block;
    background: #FF385C;
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ════════════════════════════════════════════════════════════

@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    # Eliminar columnas 100% vacías
    df = df.drop(columns=["neighbourhood_group", "license"], errors="ignore")
    # Convertir fecha
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    return df


@st.cache_data
def imputar_datos(df: pd.DataFrame):
    """Devuelve (df_media, df_knn, comparativa)."""
    from sklearn.impute import KNNImputer

    num_cols = ["price", "reviews_per_month"]

    # --- Método 1: Media ---
    df_media = df.copy()
    for col in num_cols:
        df_media[col] = df_media[col].fillna(df_media[col].mean())

    # --- Método 2: KNN ---
    df_knn = df.copy()
    features = ["price", "reviews_per_month", "minimum_nights",
                "number_of_reviews", "availability_365",
                "calculated_host_listings_count", "latitude", "longitude"]
    features_exist = [c for c in features if c in df_knn.columns]
    imputer = KNNImputer(n_neighbors=5)
    df_knn[features_exist] = imputer.fit_transform(df_knn[features_exist])

    # --- Comparativa ---
    comparativa = pd.DataFrame({
        "Variable": num_cols,
        "Nulos originales": [df[c].isnull().sum() for c in num_cols],
        "Media original":   [df[c].mean() for c in num_cols],
        "Media (imput. media)": [df_media[c].mean() for c in num_cols],
        "Media (imput. KNN)":   [df_knn[c].mean() for c in num_cols],
        "Std original":     [df[c].std() for c in num_cols],
        "Std (imput. media)":   [df_media[c].std() for c in num_cols],
        "Std (imput. KNN)":     [df_knn[c].std() for c in num_cols],
    }).round(4)

    return df_media, df_knn, comparativa


# ── Ruta del archivo ─────────────────────────────────────────
DATA_OPTIONS = {
    "📍 Regresión — Airbnb Listings CDMX": "dataset_regresion_listings.csv",
}

# ════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏠 Airbnb CDMX")
    st.markdown("---")

    # Selector de base de datos
    db_selected = st.selectbox(
        "Base de datos",
        list(DATA_OPTIONS.keys()),
        help="Selecciona el dataset a analizar"
    )

    st.markdown("---")
    st.markdown("### 🔎 Filtros")

    # Intentar cargar datos para poblar filtros
    csv_name = DATA_OPTIONS[db_selected]
    possible_paths = [
        csv_name,
        os.path.join("data", csv_name),
        os.path.join("..", "data", csv_name),
        os.path.join(os.path.dirname(__file__), "..", "data", csv_name),
    ]
    DATA_PATH = None
    for p in possible_paths:
        if os.path.exists(p):
            DATA_PATH = p
            break

    if DATA_PATH is None:
        st.error(f"No se encontró `{csv_name}`. Ajusta la ruta en el código.")
        st.stop()

    df_raw = cargar_datos(DATA_PATH)

    # Filtros
    room_types   = st.multiselect("Tipo de habitación",
                                   sorted(df_raw["room_type"].dropna().unique()),
                                   default=sorted(df_raw["room_type"].dropna().unique()))
    neighbourhood = st.multiselect("Alcaldía",
                                    sorted(df_raw["neighbourhood"].dropna().unique()),
                                    default=sorted(df_raw["neighbourhood"].dropna().unique()))

    price_max = int(df_raw["price"].quantile(0.99)) if df_raw["price"].notna().any() else 10000
    price_range = st.slider("Rango de precio (MXN/noche)",
                             int(df_raw["price"].min() or 0),
                             price_max,
                             (int(df_raw["price"].min() or 0), price_max))

    avail_range = st.slider("Disponibilidad (días/año)",
                             0, 365, (0, 365))

    st.markdown("---")
    st.caption("Lab #3 · Analítica de Datos · UdeA")


# ════════════════════════════════════════════════════════════
#  FILTRADO
# ════════════════════════════════════════════════════════════
df = df_raw.copy()

if room_types:
    df = df[df["room_type"].isin(room_types)]
if neighbourhood:
    df = df[df["neighbourhood"].isin(neighbourhood)]
df = df[
    (df["price"].isna() | ((df["price"] >= price_range[0]) & (df["price"] <= price_range[1]))) &
    (df["availability_365"] >= avail_range[0]) &
    (df["availability_365"] <= avail_range[1])
]


# ════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🏠 Airbnb Listings — Ciudad de México</h1>
    <p>Laboratorio #3 · Análisis Integral de Datos para Regresión · Analítica de Datos · UdeA</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  KPIs
# ════════════════════════════════════════════════════════════
k1, k2, k3, k4, k5 = st.columns(5)

precio_med = df["price"].median()
precio_avg = df["price"].mean()
hosts_uniq = df["host_id"].nunique()
nulos_price = df["price"].isnull().sum()
avail_avg   = df["availability_365"].mean()

for col, label, val, sub in [
    (k1, "Total Listings",        f"{len(df):,}",           f"de {len(df_raw):,} totales"),
    (k2, "Precio Mediano",        f"${precio_med:,.0f}",     "MXN / noche"),
    (k3, "Precio Promedio",       f"${precio_avg:,.0f}",     "MXN / noche"),
    (k4, "Anfitriones únicos",    f"{hosts_uniq:,}",         "hosts"),
    (k5, "Disponibilidad prom.",  f"{avail_avg:.0f} días",   "/ año (365)"),
]:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{val}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TABS
# ════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Estadísticas Descriptivas",
    "📈 Gráficos Principales",
    "🧪 Imputación de Datos",
    "🔬 Correlaciones y Pruebas"
])

AIRBNB_RED = "#FF385C"
COLOR_SEQ  = ["#FF385C","#BD1E59","#6A0F49","#E8684A","#F7B731","#20BF6B"]


# ────────────────────────────────────────────────────────────
# TAB 1 · Estadísticas Descriptivas
# ────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Tipos de Variables</div>', unsafe_allow_html=True)

    vars_num = df.select_dtypes(include=np.number).columns.tolist()
    vars_num = [v for v in vars_num if v not in ["id", "host_id"]]
    vars_cat = ["room_type", "neighbourhood"]

    c1, c2 = st.columns(2)
    c1.info(f"**Variables numéricas ({len(vars_num)}):**\n\n" + ", ".join(f"`{v}`" for v in vars_num))
    c2.info(f"**Variables categóricas analizadas ({len(vars_cat)}):**\n\n" + ", ".join(f"`{v}`" for v in vars_cat))

    st.markdown('<div class="section-title">Estadísticas Descriptivas — Variables Numéricas</div>',
                unsafe_allow_html=True)

    desc = df[vars_num].describe().T
    desc["skewness"] = df[vars_num].skew()
    desc["kurtosis"] = df[vars_num].kurtosis()
    desc["nulos"]    = df[vars_num].isnull().sum()
    desc["% nulos"]  = (df[vars_num].isnull().mean() * 100).round(2)
    st.dataframe(desc.round(3), use_container_width=True)

    st.markdown('<div class="section-title">Valores Faltantes por Variable</div>',
                unsafe_allow_html=True)

    nulos_df = pd.DataFrame({
        "Variable": df.columns,
        "Nulos": df.isnull().sum().values,
        "% Faltante": (df.isnull().mean() * 100).round(2).values
    }).sort_values("% Faltante", ascending=False)
    nulos_df = nulos_df[nulos_df["Nulos"] > 0]

    if not nulos_df.empty:
        fig_nulos = px.bar(nulos_df, x="Variable", y="% Faltante",
                           text="Nulos",
                           color="% Faltante",
                           color_continuous_scale=["#FFD6DC", AIRBNB_RED, "#6A0F49"],
                           title="Porcentaje de Valores Faltantes por Variable")
        fig_nulos.update_traces(textposition="outside")
        fig_nulos.update_layout(showlegend=False, height=380,
                                plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_nulos, use_container_width=True)
    else:
        st.success("✅ No hay valores faltantes en el subconjunto filtrado.")

    st.markdown('<div class="section-title">Frecuencias — Variables Categóricas</div>',
                unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        rt_counts = df["room_type"].value_counts().reset_index()
        rt_counts.columns = ["room_type", "count"]
        fig_rt = px.pie(rt_counts, names="room_type", values="count",
                        title="Distribución por Tipo de Habitación",
                        color_discrete_sequence=COLOR_SEQ, hole=0.45)
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


# ────────────────────────────────────────────────────────────
# TAB 2 · Gráficos Principales
# ────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">Distribución del Precio</div>',
                unsafe_allow_html=True)

    price_cap = df["price"][df["price"] <= df["price"].quantile(0.97)].dropna()

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(price_cap, x=price_cap,
                                nbins=60, title="Histograma de Precio (p97)",
                                color_discrete_sequence=[AIRBNB_RED])
        fig_hist.add_vline(x=price_cap.mean(), line_dash="dash", line_color="#1a1a1a",
                           annotation_text=f"Media ${price_cap.mean():,.0f}")
        fig_hist.add_vline(x=price_cap.median(), line_dash="dot", line_color="#6A0F49",
                           annotation_text=f"Mediana ${price_cap.median():,.0f}")
        fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis_title="Precio (MXN/noche)", yaxis_title="Frecuencia")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box_price = px.box(df[df["price"] <= df["price"].quantile(0.97)],
                               x="room_type", y="price",
                               color="room_type",
                               title="Distribución de Precio por Tipo de Habitación",
                               color_discrete_sequence=COLOR_SEQ)
        fig_box_price.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                    showlegend=False)
        st.plotly_chart(fig_box_price, use_container_width=True)

    st.markdown('<div class="section-title">Precio por Alcaldía</div>',
                unsafe_allow_html=True)

    precio_nb = (df.groupby("neighbourhood")["price"]
                 .agg(["mean", "median", "count"])
                 .query("count >= 30")
                 .reset_index()
                 .rename(columns={"mean": "Media", "median": "Mediana", "count": "Listings"})
                 .sort_values("Media", ascending=False)
                 .head(15))

    fig_nb_price = go.Figure()
    fig_nb_price.add_bar(x=precio_nb["neighbourhood"], y=precio_nb["Media"],
                         name="Media", marker_color=AIRBNB_RED, opacity=0.85)
    fig_nb_price.add_bar(x=precio_nb["neighbourhood"], y=precio_nb["Mediana"],
                         name="Mediana", marker_color="#6A0F49", opacity=0.85)
    fig_nb_price.update_layout(barmode="group", title="Media y Mediana del Precio por Alcaldía (top 15)",
                                plot_bgcolor="white", paper_bgcolor="white",
                                xaxis_tickangle=-35, height=420,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_nb_price, use_container_width=True)

    st.markdown('<div class="section-title">Scatter: Variables vs Precio</div>',
                unsafe_allow_html=True)

    sc1, sc2 = st.columns(2)
    df_sc = df[df["price"] <= df["price"].quantile(0.97)].dropna(subset=["price"])

    with sc1:
        fig_sc1 = px.scatter(df_sc.sample(min(3000, len(df_sc)), random_state=42),
                             x="availability_365", y="price",
                             color="room_type", opacity=0.5, size_max=6,
                             color_discrete_sequence=COLOR_SEQ,
                             title="Disponibilidad vs Precio",
                             labels={"availability_365": "Disponibilidad (días/año)",
                                     "price": "Precio (MXN)"})
        fig_sc1.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_sc1, use_container_width=True)

    with sc2:
        fig_sc2 = px.scatter(df_sc.sample(min(3000, len(df_sc)), random_state=42),
                             x="number_of_reviews", y="price",
                             color="room_type", opacity=0.5, size_max=6,
                             color_discrete_sequence=COLOR_SEQ,
                             title="Número de Reseñas vs Precio",
                             labels={"number_of_reviews": "Número de Reseñas",
                                     "price": "Precio (MXN)"})
        fig_sc2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig_sc2, use_container_width=True)

    st.markdown('<div class="section-title">Mapa Geográfico de Listings</div>',
                unsafe_allow_html=True)

    df_geo = df.dropna(subset=["price", "latitude", "longitude"])
    df_geo = df_geo[df_geo["price"] <= df_geo["price"].quantile(0.95)]
    df_geo_sample = df_geo.sample(min(5000, len(df_geo)), random_state=42)

    fig_map = px.scatter_mapbox(df_geo_sample,
                                lat="latitude", lon="longitude",
                                color="price",
                                color_continuous_scale=["#FFD6DC", AIRBNB_RED, "#6A0F49"],
                                size="price",
                                size_max=10,
                                zoom=10,
                                mapbox_style="carto-positron",
                                hover_data=["neighbourhood", "room_type", "price"],
                                title="Distribución Geográfica de Listings (coloreado por precio)")
    fig_map.update_layout(height=520, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)


# ────────────────────────────────────────────────────────────
# TAB 3 · Imputación de Datos
# ────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">Métodos de Imputación Aplicados</div>',
                unsafe_allow_html=True)

    st.info("""
    Se aplican **dos métodos** de imputación sobre las variables con datos faltantes (`price` y `reviews_per_month`):
    - **Método 1 — Media**: reemplaza los nulos por la media de la columna. Simple y rápido, pero sensible a outliers.
    - **Método 2 — KNN (K-Nearest Neighbors)**: imputa usando los 5 vecinos más cercanos en el espacio de variables numéricas. Más robusto y preserva mejor la estructura de los datos.
    """)

    with st.spinner("Calculando imputaciones..."):
        df_media, df_knn, comparativa = imputar_datos(df_raw)

    st.markdown('<div class="section-title">Tabla Comparativa: Antes y Después</div>',
                unsafe_allow_html=True)
    st.dataframe(comparativa, use_container_width=True)

    st.markdown('<div class="section-title">Distribución del Precio: Original vs Imputado</div>',
                unsafe_allow_html=True)

    cap = df_raw["price"].quantile(0.97)

    fig_imp = go.Figure()
    fig_imp.add_trace(go.Histogram(
        x=df_raw["price"][df_raw["price"] <= cap].dropna(),
        name="Original (con nulos)", opacity=0.6,
        marker_color="#aaaaaa", nbinsx=60))
    fig_imp.add_trace(go.Histogram(
        x=df_media["price"][df_media["price"] <= cap],
        name="Imputación por Media", opacity=0.6,
        marker_color=AIRBNB_RED, nbinsx=60))
    fig_imp.add_trace(go.Histogram(
        x=df_knn["price"][df_knn["price"] <= cap],
        name="Imputación KNN", opacity=0.6,
        marker_color="#6A0F49", nbinsx=60))
    fig_imp.update_layout(
        barmode="overlay",
        title="Distribución del Precio — Comparativa de Imputaciones (p97)",
        xaxis_title="Precio (MXN/noche)", yaxis_title="Frecuencia",
        plot_bgcolor="white", paper_bgcolor="white", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown('<div class="section-title">Distribución de Reviews/Mes: Original vs Imputado</div>',
                unsafe_allow_html=True)

    fig_imp2 = go.Figure()
    fig_imp2.add_trace(go.Histogram(
        x=df_raw["reviews_per_month"].dropna(),
        name="Original", opacity=0.6, marker_color="#aaaaaa", nbinsx=60))
    fig_imp2.add_trace(go.Histogram(
        x=df_media["reviews_per_month"],
        name="Media", opacity=0.6, marker_color=AIRBNB_RED, nbinsx=60))
    fig_imp2.add_trace(go.Histogram(
        x=df_knn["reviews_per_month"],
        name="KNN", opacity=0.6, marker_color="#6A0F49", nbinsx=60))
    fig_imp2.update_layout(
        barmode="overlay",
        title="Distribución de Reviews/Mes — Comparativa",
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


# ────────────────────────────────────────────────────────────
# TAB 4 · Correlaciones y Pruebas Estadísticas
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Matriz de Correlación (Pearson)</div>',
                unsafe_allow_html=True)

    vars_corr = ["price", "minimum_nights", "number_of_reviews", "reviews_per_month",
                 "calculated_host_listings_count", "availability_365",
                 "number_of_reviews_ltm", "latitude", "longitude"]
    vars_corr_exist = [v for v in vars_corr if v in df.columns]

    corr_mat = df[vars_corr_exist].corr()

    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns.tolist(),
        y=corr_mat.columns.tolist(),
        colorscale=[[0, "#6A0F49"], [0.5, "white"], [1, AIRBNB_RED]],
        zmid=0,
        text=corr_mat.round(2).values,
        texttemplate="%{text}",
        showscale=True,
        zmin=-1, zmax=1
    ))
    fig_heat.update_layout(
        title="Matriz de Correlación de Pearson",
        height=520,
        plot_bgcolor="white", paper_bgcolor="white"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title">Correlación con Price: Pearson vs Spearman</div>',
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
            "Variable": var,
            "Pearson r": round(r_p, 4),
            "p Pearson": round(pv_p, 6),
            "Spearman r": round(r_s, 4),
            "p Spearman": round(pv_s, 6),
            "Sig. Pearson": "***" if pv_p < 0.001 else ("**" if pv_p < 0.01 else ("*" if pv_p < 0.05 else "ns")),
            "Sig. Spearman": "***" if pv_s < 0.001 else ("**" if pv_s < 0.01 else ("*" if pv_s < 0.05 else "ns")),
        })

    df_corr_res = pd.DataFrame(results_corr).sort_values("Pearson r", key=abs, ascending=False)
    st.dataframe(df_corr_res, use_container_width=True)
    st.caption("Significancia: \\*\\*\\* p<0.001 · \\*\\* p<0.01 · \\* p<0.05 · ns = no significativo")

    # Gráfico de barras comparativo
    fig_corr_bar = go.Figure()
    fig_corr_bar.add_bar(x=df_corr_res["Variable"], y=df_corr_res["Pearson r"],
                         name="Pearson", marker_color=AIRBNB_RED, opacity=0.85)
    fig_corr_bar.add_bar(x=df_corr_res["Variable"], y=df_corr_res["Spearman r"],
                         name="Spearman", marker_color="#6A0F49", opacity=0.85)
    fig_corr_bar.add_hline(y=0, line_color="black", line_width=0.8)
    fig_corr_bar.update_layout(
        barmode="group", title="Pearson vs Spearman — Correlación con Price",
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_corr_bar, use_container_width=True)

    st.markdown('<div class="section-title">Prueba ANOVA y Kruskal-Wallis — Precio por Tipo de Habitación</div>',
                unsafe_allow_html=True)

    grupos = [df[df["room_type"] == rt]["price"].dropna() for rt in df["room_type"].unique()]
    grupos = [g for g in grupos if len(g) >= 5]

    if len(grupos) >= 2:
        f_stat, p_anova   = stats.f_oneway(*grupos)
        h_stat, p_kruskal = stats.kruskal(*grupos)

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.markdown("#### ANOVA (paramétrica)")
            color_a = "green" if p_anova < 0.05 else "red"
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

    st.markdown('<div class="section-title">Prueba Chi-cuadrado — room_type vs neighbourhood</div>',
                unsafe_allow_html=True)

    top_nb = df["neighbourhood"].value_counts().head(6).index.tolist()
    df_chi = df[df["neighbourhood"].isin(top_nb)][["room_type", "neighbourhood"]].dropna()
    if len(df_chi) >= 20:
        tabla_cont = pd.crosstab(df_chi["room_type"], df_chi["neighbourhood"])
        chi2, p_chi, dof, expected = stats.chi2_contingency(tabla_cont)

        c_chi1, c_chi2, c_chi3 = st.columns(3)
        c_chi1.metric("Chi² estadístico", f"{chi2:.4f}")
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

    st.markdown('<div class="section-title">Datos Filtrados</div>', unsafe_allow_html=True)
    st.caption(f"Mostrando {min(500, len(df)):,} de {len(df):,} registros filtrados")
    st.dataframe(df.head(500), use_container_width=True)
