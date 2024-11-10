# Importar bibliotecas necesarias
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargamos el conjunto de datos
df = pd.read_csv("https://raw.githubusercontent.com/Fibovin/des_modelos_1/refs/heads/main/Student_Performance.csv")

# Codificamos la variable categórica
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

# Dividimos el conjunto de datos en características y variable objetivo
X = df.drop(columns=['Performance Index'])
y = df['Performance Index']

# Dividimos el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuramos el servidor MLflow
mlflow.set_tracking_uri("http://0.0.0.0:8050")

# Configuramos el experimento de MLflow
mlflow.set_experiment("EXPERIMENTOS CON SVR")

# Definimos y ejecutamos el experimento con MLflow para el modelo SVR
with mlflow.start_run(run_name="Reg_SVR Experimento_1"):
    # Creamos y entrenamos el modelo SVR con kernel lineal
    svr_model = SVR(kernel="linear")
    svr_model.fit(X_train, y_train)

    # Realizamos predicciones en el conjunto de prueba
    y_test_pred_svr = svr_model.predict(X_test)

    # Calculamos y registramos las métricas para el conjunto de prueba
    r2_test_svr = r2_score(y_test, y_test_pred_svr)
    mse_test_svr = mean_squared_error(y_test, y_test_pred_svr)
    mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
    
    mlflow.log_metric("r2", r2_test_svr)
    mlflow.log_metric("mse", mse_test_svr)
    mlflow.log_metric("mae", mae_test_svr)

    # Registramos el modelo SVR en MLflow
    mlflow.sklearn.log_model(svr_model, "svr-linear-model")

    # Imprimimos las métricas para verificar
    print("Conjunto de Prueba - SVR:")
    print(f"R²: {r2_test_svr}")
    print(f"Mean Squared Error: {mse_test_svr}")
    print(f"Mean Absolute Error: {mae_test_svr}")
