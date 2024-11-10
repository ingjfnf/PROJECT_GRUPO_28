# Importar bibliotecas necesarias
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression, Lasso
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
mlflow.set_experiment("EXPERIMENTOS CON REGRESIÓN LINEAL")

# Definimos y ejecutamos el experimento con MLflow para la regresión lineal
# Experimento con Lasso Regression (Experimento 3)
with mlflow.start_run(run_name="REGRESION LASSO Experimento_3"):
    # Creamos y entrenamos el modelo Lasso
    lasso_model = Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    # Realizamos predicciones en el conjunto de prueba
    y_test_pred_lasso = lasso_model.predict(X_test)

    # Calculamos y registramos las métricas para el conjunto de prueba
    r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
    mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
    mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
    
    mlflow.log_metric("r2", r2_test_lasso)
    mlflow.log_metric("mse", mse_test_lasso)
    mlflow.log_metric("mae", mae_test_lasso)

    # Registramos el modelo Lasso en MLflow
    mlflow.sklearn.log_model(lasso_model, "lasso-regression-model")

    # Imprimimos las métricas para verificar
    print("\nConjunto de Prueba - Lasso Regression:")
    print(f"R²: {r2_test_lasso}")
    print(f"Mean Squared Error: {mse_test_lasso}")
    print(f"Mean Absolute Error: {mae_test_lasso}")

