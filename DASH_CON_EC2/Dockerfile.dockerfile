FROM python:3.9-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos todos los archivos al contenedor
COPY . /app

# Instalamos las dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# Creamos las carpetas necesarias
RUN mkdir -p data models

# Ejecutamos el preprocesamiento y entrenamiento
RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

# Exponemos el puerto que usa la aplicación
EXPOSE 8001

# Comando para iniciar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8001", "--server.address=0.0.0.0"]
