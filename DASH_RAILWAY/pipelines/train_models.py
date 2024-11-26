import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def cargar_datos():
    # Descarga y procesa los datos en memoria
    url = "https://raw.githubusercontent.com/Fibovin/des_modelos_1/refs/heads/main/Student_Performance.csv"
    datos = pd.read_csv(url)
    datos['Extracurricular Activities'] = datos['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    return datos

def entrenar_modelos(X, y):
    # Define y entrena los modelos, guardándolos como .pkl
    modelos = {
        "linear_model.pkl": LinearRegression(),
        "svr_model.pkl": SVR(kernel="linear"),
        "decision_tree.pkl": DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    }
    for nombre, modelo in modelos.items():
        modelo.fit(X, y)
        with open(f"models/{nombre}", "wb") as archivo:
            pickle.dump(modelo, archivo)
        print(f"Modelo {nombre} guardado exitosamente.")

if __name__ == "__main__":
    # Entrena los modelos usando los datos procesados
    datos = cargar_datos()
    X = datos[['Hours Studied', 'Previous Scores', 'Extracurricular Activities', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = datos['Performance Index']
    entrenar_modelos(X, y)
