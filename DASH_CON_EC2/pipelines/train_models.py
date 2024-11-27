import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

def cargar_datos():
    # Cargamos los datos procesados desde el archivo CSV
    return pd.read_csv("data/Student_Performance_Procesado.csv")

def entrenar_modelos(X_entrenamiento, y_entrenamiento):
    # Diccionario de modelos
    modelos = {
        "linear_model.pkl": LinearRegression(),
        "svr_model.pkl": SVR(kernel="linear"),
        "decision_tree.pkl": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    }

    # Entrenamos cada modelo y guardar como archivo .pkl
    for nombre, modelo in modelos.items():
        modelo.fit(X_entrenamiento, y_entrenamiento)
        with open(f"models/{nombre}", "wb") as archivo:
            pickle.dump(modelo, archivo)
        print(f"Modelo {nombre} guardado exitosamente.")

if __name__ == "__main__":
    # Cargamos los datos procesados
    datos = cargar_datos()
    
    # Seleccionamos las columnas relevantes para X e y
    X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = datos['Performance Index']
    
    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamos y guardamos los modelos
    entrenar_modelos(X_entrenamiento, y_entrenamiento)
