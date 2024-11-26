FROM python:3.9-slim

# Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar todos los archivos desde el directorio local al contenedor
COPY . /app

# Actualizar pip e instalar las dependencias desde requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Crear las carpetas necesarias para datos y modelos
RUN mkdir -p data models

# Ejecutar el preprocesamiento de datos y el entrenamiento de modelos
RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

# Exponer el puerto que usará la aplicación en el contenedor
EXPOSE 8000

# Comando de inicio para ejecutar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
