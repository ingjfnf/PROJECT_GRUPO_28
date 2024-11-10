# Importamos bibliotecas necesarias
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
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
mlflow.set_experiment("EXPERIMENTOS CON MODELO DE ARBOL")

with mlflow.start_run(run_name="Regresion ARBOL Experimento_2"):
    # Creamos el modelo con los mejores hiperparámetros
    optimized_tree_model = DecisionTreeRegressor(max_depth=10, min_samples_leaf=4, min_samples_split=10, random_state=42)
    optimized_tree_model.fit(X_train, y_train)

    # Realizamos predicciones en el conjunto de prueba
    y_test_pred_optimized = optimized_tree_model.predict(X_test)

    # Registramos el modelo optimizado en MLflow
    mlflow.sklearn.log_model(optimized_tree_model, "decision-tree-model")	

    # Calculamos y registramos las métricas para el conjunto de prueba
    r2_test_optimized = r2_score(y_test, y_test_pred_optimized)
    mse_test_optimized = mean_squared_error(y_test, y_test_pred_optimized)
    mae_test_optimized = mean_absolute_error(y_test, y_test_pred_optimized)
    
    mlflow.log_metric("r2_score", r2_test_optimized)
    mlflow.log_metric("mse", mse_test_optimized)
    mlflow.log_metric("mae", mae_test_optimized)


    # Imprimimos las métricas para verificar
    print("Conjunto de Prueba:")
    print(f"R²: {r2_test_optimized}")
    print(f"Mean Squared Error: {mse_test_optimized}")
    print(f"Mean Absolute Error: {mae_test_optimized}")
