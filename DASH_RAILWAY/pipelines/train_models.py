import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def cargar_datos():
    # Descarga y procesa los datos en memoria
    url = "https://raw.githubusercontent.com/Fibovin/des_modelos_1/refs/heads/main/Student_Performance.csv"
    datos = pd.read_csv(url)
    datos['Extracurricular Activities'] = datos['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return datos

def entrenar_modelos(X_entrenamiento, y_entrenamiento):
    # Define y entrena los modelos, guardándolos como .pkl
    modelos = {
        "linear_model.pkl": LinearRegression(),
        "svr_model.pkl": SVR(kernel="linear"),
        "decision_tree.pkl": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X_entrenamiento, y_entrenamiento)
        with open(f"models/{nombre}", "wb") as archivo:
            pickle.dump(modelo, archivo)
        print(f"Modelo {nombre} guardado exitosamente.")

if __name__ == "__main__":
    # Cargamos los datos
    datos = cargar_datos()

    # Seleccionamos las variables predictoras y la variable objetivo
    X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = datos['Performance Index']

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, _, y_entrenamiento, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamos los modelos utilizando solo los datos de entrenamiento
    entrenar_modelos(X_entrenamiento, y_entrenamiento)

