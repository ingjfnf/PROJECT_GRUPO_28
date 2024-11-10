import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
mlflow.set_experiment("EXPERIMENTO_MEJORES_VERSIONES")

# Definimos y ejecutamos el experimento con MLflow para la regresión lineal
with mlflow.start_run(run_name="MEJOR MODELO REGRESION LINEAL"):
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_test_pred = linear_model.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    mlflow.log_metric("r2", r2_test)
    mlflow.log_metric("mse", mse_test)
    mlflow.log_metric("mae", mae_test)
    mlflow.sklearn.log_model(linear_model, "linear-regression-model")
    print("Regresión Lineal:")
    print(f"R²: {r2_test}, MSE: {mse_test}, MAE: {mae_test}")

# Definimos y ejecutamos el experimento con MLflow para el modelo SVR
with mlflow.start_run(run_name="MEJOR MODELO SVR"):
    svr_model = SVR(kernel="linear")
    svr_model.fit(X_train, y_train)
    y_test_pred_svr = svr_model.predict(X_test)
    r2_test_svr = r2_score(y_test, y_test_pred_svr)
    mse_test_svr = mean_squared_error(y_test, y_test_pred_svr)
    mae_test_svr = mean_absolute_error(y_test, y_test_pred_svr)
    
    mlflow.log_metric("r2", r2_test_svr)
    mlflow.log_metric("mse", mse_test_svr)
    mlflow.log_metric("mae", mae_test_svr)
    mlflow.sklearn.log_model(svr_model, "svr-linear-model")
    print("SVR:")
    print(f"R²: {r2_test_svr}, MSE: {mse_test_svr}, MAE: {mae_test_svr}")

# Definimos y ejecutamos el experimento con MLflow para el modelo de árbol de decisión optimizado
with mlflow.start_run(run_name="MEJOR MODELO ARBOL DECISION"):
    optimized_tree_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    optimized_tree_model.fit(X_train, y_train)
    y_test_pred_optimized = optimized_tree_model.predict(X_test)
    r2_test_optimized = r2_score(y_test, y_test_pred_optimized)
    mse_test_optimized = mean_squared_error(y_test, y_test_pred_optimized)
    mae_test_optimized = mean_absolute_error(y_test, y_test_pred_optimized)
    
    mlflow.log_metric("r2", r2_test_optimized)
    mlflow.log_metric("mse", mse_test_optimized)
    mlflow.log_metric("mae", mae_test_optimized)
    mlflow.sklearn.log_model(optimized_tree_model, "decision-tree-model")
    print("Árbol de Decisión:")
    print(f"R²: {r2_test_optimized}, MSE: {mse_test_optimized}, MAE: {mae_test_optimized}")
