import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

# Configuración de la página
st.set_page_config(
    page_title="Chicago Crime Analytics | Big Data",
    page_icon="🚓",
    layout="wide"
)

st.markdown("""
    <style>
    div[data-testid="stMetricValue"] { color: #A855F7; font-size: 2rem; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_engine():
    return create_engine("postgresql+psycopg2://admin:adminpassword@postgres:5432/chicago_crime_db")

@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    df_temp = pd.read_sql("SELECT * FROM fact_tendencia_temporal", engine)
    df_hora = pd.read_sql("SELECT * FROM fact_patron_horario", engine)
    df_dist = pd.read_sql("SELECT * FROM fact_resumen_distrito", engine)
    df_mapa = pd.read_sql("SELECT * FROM fact_mapa_puntos", engine)
    return df_temp, df_hora, df_dist, df_mapa

st.title("🚓 Análisis Masivo de Criminalidad en Chicago (2001 - 2022)")
st.markdown("**Arquitectura Batch:** Apache Spark (PySpark) ➔ PostgreSQL ➔ Streamlit")

try:
    df_temp, df_hora, df_dist, df_mapa = load_data()
except Exception as e:
    st.error(f"Error conectando a PostgreSQL: {e}")
    st.stop()

# --- BARRA LATERAL (FILTROS) ---
st.sidebar.header("🔎 Filtros de Exploración")

anios_disp = sorted(df_temp["anio"].unique())
rango_anios = st.sidebar.slider(
    "Rango de Años",
    min_value=int(min(anios_disp)),
    max_value=int(max(anios_disp)),
    value=(int(min(anios_disp)), int(max(anios_disp)))
)

top_delitos = df_temp.groupby("tipo_delito")["total_delitos"].sum().nlargest(15).index.tolist()
delitos_sel = st.sidebar.multiselect(
    "Tipos de Delito (Top 15)",
    options=sorted(df_temp["tipo_delito"].unique()),
    default=top_delitos[:6]
)

if not delitos_sel:
    st.warning("Selecciona al menos un tipo de delito en la barra lateral.")
    st.stop()

# Filtrado de DataFrames
mask_temp = (df_temp["anio"].between(rango_anios[0], rango_anios[1])) & (df_temp["tipo_delito"].isin(delitos_sel))
f_temp = df_temp[mask_temp]

mask_dist = (df_dist["anio"].between(rango_anios[0], rango_anios[1])) & (df_dist["tipo_delito"].isin(delitos_sel))
f_dist = df_dist[mask_dist]

f_hora = df_hora[df_hora["tipo_delito"].isin(delitos_sel)]
f_mapa = df_mapa[df_mapa["tipo_delito"].isin(delitos_sel)]

# --- KPIS PRINCIPALES ---
total_casos = int(f_temp["total_delitos"].sum())
total_arrestos = int(f_temp["total_arrestos"].sum())
tasa_arresto = (total_arrestos / total_casos * 100) if total_casos > 0 else 0
total_domesticos = int(f_temp["total_domesticos"].sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Incidentes Analizados", f"{total_casos:,}")
c2.metric("Total Arrestos Efectivos", f"{total_arrestos:,}")
c3.metric("Tasa de Arresto", f"{tasa_arresto:.2f}%")
c4.metric("Incidentes Domésticos", f"{total_domesticos:,}")

st.divider()

# --- PESTAÑAS DE ANÁLISIS ---
tab1, tab2, tab3 = st.tabs(["📈 Patrones Temporales", "🗺️ Patrones Geográficos", "📊 Comparativa por Delito"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Evolución Anual de Delitos")
        evolucion = f_temp.groupby(["anio", "tipo_delito"], as_index=False)["total_delitos"].sum()
        fig_line = px.line(
            evolucion, x="anio", y="total_delitos", color="tipo_delito",
            markers=True, template="plotly_dark",
            labels={"anio": "Año", "total_delitos": "Incidentes", "tipo_delito": "Delito"}
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("Mapa de Calor: Día de la Semana vs. Hora")
        heatmap_data = f_hora.groupby(["dia_semana_num", "dia_semana", "hora"], as_index=False)["total_delitos"].sum()
        heatmap_pivot = heatmap_data.pivot(index="dia_semana", columns="hora", values="total_delitos").fillna(0)
        orden_dias = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap_pivot = heatmap_pivot.reindex([d for d in orden_dias if d in heatmap_pivot.index])
        fig_heat = px.imshow(
            heatmap_pivot, aspect="auto", color_continuous_scale="Purples",
            labels=dict(x="Hora del Día (0-23h)", y="Día", color="Delitos"),
            template="plotly_dark"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

with tab2:
    col_g1, col_g2 = st.columns([1, 1.3])
    with col_g1:
        st.subheader("Top Distritos Policiales con Mayor Incidencia")
        top_dist = f_dist.groupby("distrito", as_index=False)["total_delitos"].sum().sort_values("total_delitos", ascending=False).head(12)
        top_dist["distrito"] = "Distrito " + top_dist["distrito"].astype(str)
        fig_bar_dist = px.bar(
            top_dist, x="total_delitos", y="distrito", orientation="h",
            color="total_delitos", color_continuous_scale="Purples",
            template="plotly_dark", labels={"total_delitos": "Total Delitos", "distrito": "Distrito"}
        )
        fig_bar_dist.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar_dist, use_container_width=True)

    with col_g2:
        st.subheader("Distribución Georreferenciada en Chicago")
        if not f_mapa.empty:
            muestra_viz = f_mapa.sample(min(3000, len(f_mapa)), random_state=42).rename(
                columns={"latitud": "lat", "longitud": "lon"}
            )
            st.map(muestra_viz, latitude="lat", longitude="lon", color="#A855F7", zoom=10)
        else:
            st.info("No hay coordenadas disponibles para los filtros seleccionados.")

with tab3:
    st.subheader("Volumen de Incidentes vs. Efectividad de Arresto")
    resumen_delito = f_temp.groupby("tipo_delito", as_index=False).agg(
        Incidentes=("total_delitos", "sum"),
        Arrestos=("total_arrestos", "sum")
    )
    resumen_delito["Tasa_Arresto_%"] = (resumen_delito["Arrestos"] / resumen_delito["Incidentes"] * 100).round(2)
    resumen_delito = resumen_delito.sort_values("Incidentes", ascending=False)
    st.dataframe(resumen_delito, use_container_width=True, hide_index=True)