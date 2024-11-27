import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import pickle

# Establecemos el título principal del tablero
st.title("Tablero de Rendimiento Estudiantil")

# Proporcionamos una breve explicación del tablero
st.write("""
Este tablero presenta un análisis de rendimiento estudiantil en función de varias características como las horas de estudio, puntajes previos, actividades extracurriculares, horas de sueño y cuestionarios practicados.
Utilizando distintos modelos de regresión, se predice un índice de rendimiento (Performance Index) basado en estas características.
""")

# Función para cargar los datos procesados
def cargar_datos():
    # URL del dataset original
    url = "https://raw.githubusercontent.com/Fibovin/des_modelos_1/refs/heads/main/Student_Performance.csv"
    datos = pd.read_csv(url)
    datos['Extracurricular Activities'] = datos['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return datos

# Función para cargar los modelos entrenados
def cargar_modelos():
    modelos = {
        "Regresión Lineal": pickle.load(open("models/linear_model.pkl", "rb")),
        "SVR": pickle.load(open("models/svr_model.pkl", "rb")),
        "Árbol de Decisión": pickle.load(open("models/decision_tree.pkl", "rb"))
    }
    return modelos

# Cargamos los datos
datos = cargar_datos()

# Dividir los datos en características (X) y variable objetivo (y)
X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
y = datos['Performance Index']

# Dividimos en conjunto de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

# Cargamos los modelos entrenados
modelos = cargar_modelos()

# Calculamos las métricas de cada modelo en el conjunto de prueba
metricas = []
for nombre, modelo in modelos.items():
    y_predicho = modelo.predict(X_prueba)
    mse = mean_squared_error(y_prueba, y_predicho)
    mae = mean_absolute_error(y_prueba, y_predicho)
    r2 = r2_score(y_prueba, y_predicho)
    metricas.append({"Modelo": nombre, "MSE": mse, "MAE": mae, "R²": r2})

st.markdown(
    """
    <h2 style='text-align: center;'>Comparación de Métricas de <span style="display: block; text-align: center;">Modelos</span></h2>
    """,
    unsafe_allow_html=True
)

# Presentamos la tabla de comparación de métricas en el centro
with st.columns([1, 2, 1])[1]:
    st.dataframe(pd.DataFrame(metricas))

# Identificamos el mejor modelo basado en el MSE y el R²
mejor_modelo_info = pd.DataFrame(metricas).sort_values(by=["MSE", "R²"], ascending=[True, False]).iloc[0]
nombre_mejor_modelo = mejor_modelo_info["Modelo"]

# Mostramos la sección "Mejor Modelo"
st.markdown(
    """
    <h2 style='text-align: center;'>Mejor Modelo</h2>
    """,
    unsafe_allow_html=True
)

# Mejor modelo con subrayado sin centrar
st.markdown(
    f"""
    <p style='font-size: 16px;'><u>El mejor modelo es: <strong>{nombre_mejor_modelo}</strong></u></p>
    """,
    unsafe_allow_html=True
)

st.write("Este modelo se selecciona como el mejor porque tiene el menor **MSE** (Error Cuadrático Medio) "
         "y el mayor **R²** (Coeficiente de Determinación), lo cual indica una mejor capacidad de predicción "
         "y una mayor explicación de la variabilidad en los datos.")

# Graficamos Predicciones vs Valores Reales y el Histograma de Residuales para el mejor modelo
modelo_mejor = modelos[nombre_mejor_modelo]
y_predicho_mejor = modelo_mejor.predict(X_prueba)
residuales_mejor = y_prueba - y_predicho_mejor

st.subheader(f"Comportamiento del Mejor Modelo: {nombre_mejor_modelo}")
col1, col2 = st.columns(2)
with col2:
    st.write("**Predicciones vs. Valores Reales**")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(y_prueba, y_predicho_mejor, color="green", alpha=0.5, label="Predicciones (Puntos)")
    ax1.plot([y_prueba.min(), y_prueba.max()], [y_prueba.min(), y_prueba.max()], linestyle='--', lw=2, color="red", label="Línea Ideal (Y=X)")
    ax1.set_xlabel("Valores Reales")
    ax1.set_ylabel("Predicciones")
    ax1.set_title(f"Predicciones vs Valores Reales para {nombre_mejor_modelo}")
    ax1.legend()
    st.pyplot(fig1)

with col1:
    st.write("**Histograma de Residuales**")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.hist(residuales_mejor, bins=20, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Residuales")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de los Residuales")
    st.pyplot(fig2)

# Explicamos cada modelo utilizado
descripcion_modelos = {
    "Regresión Lineal": "Modelo simple que asume una relación lineal entre las variables independientes y la variable dependiente.",
    "SVR": "Modelo de regresión basado en vectores soporte, útil para relaciones no lineales.",
    "Árbol de Decisión": "Modelo basado en divisiones jerárquicas de los datos, útil para relaciones complejas."
}

# Creamos un formulario en la barra lateral para realizar predicciones
with st.sidebar.form("prediction_form"):
    st.header("Predicción Personalizada")
    seleccion_modelo = st.selectbox("Modelo", ["Regresión Lineal", "SVR", "Árbol de Decisión"])

    st.subheader("Ingrese los Parámetros para la Predicción")
    horas_estudio = st.number_input("Horas de Estudio", min_value=0, max_value=10, value=5)
    puntaje_previo = st.number_input("Puntaje Previo", min_value=0, max_value=100, value=50)
    horas_sueno = st.number_input("Horas de Sueño", min_value=0, max_value=12, value=7)
    actividad_extra = st.selectbox("Actividades Extracurriculares", options=["No", "Sí"])
    cuestionarios_practicados = st.number_input("Cuestionarios Practicados", min_value=0, max_value=10, value=5)

    valor_actividad_extra = 1 if actividad_extra == "Sí" else 0

    submitted = st.form_submit_button("Evaluar")
    if submitted:
        entrada = pd.DataFrame([[horas_estudio, puntaje_previo, valor_actividad_extra, horas_sueno, cuestionarios_practicados]],
                               columns=X.columns)
        modelo_seleccionado = modelos[seleccion_modelo]
        prediccion = modelo_seleccionado.predict(entrada)
        st.sidebar.write(f"Predicción del Índice de Rendimiento: {prediccion[0]:.2f}")

        st.sidebar.write("Explicación del Modelo:")
        st.sidebar.write(descripcion_modelos[seleccion_modelo])
