import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Título principal del tablero
st.title("Tablero de Rendimiento Estudiantil")

# Breve explicación del tablero
st.write("""
Este tablero presenta un análisis de rendimiento estudiantil en función de varias características como las horas de estudio, horas de sueño, actividades extracurriculares y cuestionarios practicados. 
         Utilizando distintos modelos de regresión, se predice un índice de rendimiento (Performance Index) basado en estas características. 
""")

# Cargar y preparar datos
datos = pd.read_csv("Student_Performance.csv")
datos['Extracurricular Activities'] = LabelEncoder().fit_transform(datos['Extracurricular Activities'])
X = datos[['Hours Studied', 'Sleep Hours', 'Extracurricular Activities', 'Sample Question Papers Practiced']]
y = datos['Performance Index']
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=0)

# Selección del modelo
st.sidebar.header("Seleccione el Modelo")
seleccion_modelo = st.sidebar.selectbox("Modelo", ["Regresión Lineal", "Árbol de Decisión", "Bosque Aleatorio", "SVR", "Gradient Boosting", "XGBoost"])

# Configuramos el modelo seleccionado
if seleccion_modelo == "Regresión Lineal":
    modelo = LinearRegression()
elif seleccion_modelo == "Árbol de Decisión":
    modelo = DecisionTreeRegressor(random_state=0)
elif seleccion_modelo == "Bosque Aleatorio":
    modelo = RandomForestRegressor(random_state=0)
elif seleccion_modelo == "SVR":
    modelo = SVR()
elif seleccion_modelo == "Gradient Boosting":
    modelo = GradientBoostingRegressor(random_state=0)
else:
    modelo = xgb.XGBRegressor(objective='reg:squarederror', random_state=0)

# Entrenamos el modelo y predecir
modelo.fit(X_entrenamiento, y_entrenamiento)
y_predicho = modelo.predict(X_prueba)
residuales = y_prueba - y_predicho  # Calculamos residuales para el análisis

#Creamos las cajitas para que el usuario ingrese los valores para las variables independientes
st.sidebar.subheader("Ingrese los Parámetros para la Predicción")
horas_estudio = st.sidebar.number_input("Horas de Estudio", min_value=0, max_value=10, value=5)
horas_sueno = st.sidebar.number_input("Horas de Sueño", min_value=0, max_value=12, value=7)
actividad_extra = st.sidebar.selectbox("Actividades Extracurriculares", options=["No", "Sí"])
cuestionarios_practicados = st.sidebar.number_input("Cuestionarios Practicados", min_value=0, max_value=10, value=3)
valor_actividad_extra = 1 if actividad_extra == "Sí" else 0
# Mostramos predicción para valores específicos ingresados por el usuario
prediccion = modelo.predict([[horas_estudio, horas_sueno, valor_actividad_extra, cuestionarios_practicados]])

# Mostramos la predicción con el modelo seleccionado
st.sidebar.write(f"Predicción de Performance Index ({seleccion_modelo}): {prediccion[0]:.2f}")

# Gráficamos de Predicciones vs Valores Reales (a la derecha)
st.subheader(f"Comportamiento del Modelo: {seleccion_modelo}")
col1, col2 = st.columns(2)
with col2:
    st.write("**Predicciones vs. Valores Reales**")
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.scatter(y_prueba, y_predicho, color="green", alpha=0.5, label="Predicciones (Puntos)")
    ax1.plot([y_prueba.min(), y_prueba.max()], [y_prueba.min(), y_prueba.max()], 'k--', lw=2, color="red", label="Línea Ideal (Y=X)")
    ax1.set_xlabel("Valores Reales")
    ax1.set_ylabel("Predicciones")
    ax1.set_title(f"Predicciones vs. Valores Reales para {seleccion_modelo}")
    ax1.legend()
    st.pyplot(fig1)

# Gráficamos de Histograma de Residuales (a la izquierda)
with col1:
    st.write("**Histograma de Residuales**")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    residuales = y_prueba - y_predicho
    ax2.hist(residuales, bins=20, color="skyblue", edgecolor="black")
    ax2.set_xlabel("Residuales")
    ax2.set_ylabel("Frecuencia")
    ax2.set_title("Distribución de los Residuales")
    st.pyplot(fig2)

# Tabla comparativa de métricas
st.subheader("Comparación de Métricas de Modelos")
modelos = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=0),
    "Bosque Aleatorio": RandomForestRegressor(random_state=0),
    "SVR": SVR(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=0),
    "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=0)
}

metricas = []
for nombre, modelo in modelos.items():
    modelo.fit(X_entrenamiento, y_entrenamiento)
    y_predicho = modelo.predict(X_prueba)
    mse = mean_squared_error(y_prueba, y_predicho)
    mae = mean_absolute_error(y_prueba, y_predicho)
    r2 = r2_score(y_prueba, y_predicho)
    metricas.append({"Modelo": nombre, "MSE": mse, "MAE": mae, "R²": r2})

# Creamos DataFrame y mostrarlo en Streamlit
metricas_df = pd.DataFrame(metricas)
st.dataframe(metricas_df)

# Identificamos el mejor modelo basado en MSE y R²
mejor_modelo = metricas_df.sort_values(by=["MSE", "R²"], ascending=[True, False]).iloc[0]
nombre_mejor_modelo = mejor_modelo["Modelo"]

# Mostramos el mejor modelo con explicación
st.subheader("Mejor Modelo")
st.write(f"El mejor modelo es: **{nombre_mejor_modelo}**")
st.write("Este modelo se selecciona como el mejor porque tiene el menor **MSE** (Error Cuadrático Medio) "
         "y el mayor **R²** (Coeficiente de Determinación), lo cual indica una mejor capacidad de predicción "
         "y una mayor explicación de la variabilidad en los datos.")
