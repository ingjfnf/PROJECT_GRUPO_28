FROM python:3.9-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos todos los archivos al contenedor
COPY . /app

# Instalamos dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt


# Crear carpetas las necesarias
RUN mkdir -p data models


# copiamos los carpeta necesarias  al contenedor para poder ejecutar el modelo el dataset ya procesado y los pkl de los modelos
COPY data /app/data
COPY models /app/models


# Ejecutar preprocesamiento y entrenamiento
RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

# Exponer el puerto para Railway
EXPOSE 8000

# Comando para iniciar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]