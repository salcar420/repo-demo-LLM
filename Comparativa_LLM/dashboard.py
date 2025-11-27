import streamlit as st
import pandas as pd
import plotly.express as px

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Comparativa LLM Local", page_icon="🤖", layout="wide")

st.title(" Dashboard de Rendimiento: LLMs Locales")
st.markdown("Comparativa visual de velocidad, latencia y consumo de recursos usando **Ollama**.")

# --- CARGAR DATOS ---
archivo_csv = 'benchmark_pro.csv'

try:
    df = pd.read_csv(archivo_csv)
    
    # --- METRICAS PRINCIPALES (KPIs) ---
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    # Calculamos al ganador de velocidad promedio
    velocidad_promedio = df.groupby('Modelo')['Tokens_Seg(t/s)'].mean().sort_values(ascending=False)
    ganador_vel = velocidad_promedio.index[0]
    max_vel = velocidad_promedio.iloc[0]

    # Calculamos al ganador de latencia (el más bajo es mejor)
    latencia_promedio = df.groupby('Modelo')['Latencia_TTFT(s)'].mean().sort_values(ascending=True)
    ganador_lat = latencia_promedio.index[0]
    min_lat = latencia_promedio.iloc[0]

    with col1:
        st.metric(label="🏆 Modelo Más Rápido (Avg)", value=ganador_vel, delta=f"{max_vel:.2f} t/s")
    with col2:
        st.metric(label="⚡ Mejor Respuesta (Latencia)", value=ganador_lat, delta=f"{min_lat:.2f} s", delta_color="inverse")
    with col3:
        st.metric(label="🧪 Total Pruebas", value=len(df))

    st.divider()

    # --- GRÁFICOS INTERACTIVOS ---
    
    # 1. Gráfico de Velocidad por Categoría
    st.subheader("🚀 Velocidad de Generación (Tokens/segundo)")
    fig_vel = px.bar(
        df, 
        x="Modelo", 
        y="Tokens_Seg(t/s)", 
        color="Categoria", 
        barmode="group",
        text_auto='.2s',
        title="¿Quién escribe más rápido en cada tarea?",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_vel, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # 2. Gráfico de Latencia
        st.subheader("⏱️ Latencia (Tiempo de Pensado)")
        fig_lat = px.bar(
            df, 
            x="Modelo", 
            y="Latencia_TTFT(s)", 
            color="Modelo",
            title="¿Cuánto tardan en empezar a escribir?",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_lat, use_container_width=True)

    with col_b:
        # 3. Gráfico de RAM (si los datos son coherentes)
        st.subheader("💾 Impacto en Memoria RAM")
        # Filtramos valores negativos pequeños que son ruido de medición
        df['RAM_Usada(MB)'] = df['RAM_Usada(MB)'].apply(lambda x: x if x > 0 else 0)
        
        fig_ram = px.scatter(
            df, 
            x="Modelo", 
            y="RAM_Usada(MB)", 
            size="Tokens_Total", 
            color="Categoria",
            title="Consumo de Memoria vs Longitud de Respuesta"
        )
        st.plotly_chart(fig_ram, use_container_width=True)

    # --- TABLA DE DATOS BRUTOS ---
    with st.expander("📂 Ver Tabla de Datos Completa"):
        st.dataframe(df, use_container_width=True)

except FileNotFoundError:
    st.error(f"⚠️ No se encontró el archivo '{archivo_csv}'. ¡Ejecuta primero el script 'benchmark.py'!")