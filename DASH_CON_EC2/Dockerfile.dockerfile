# Usamos una imagen base ligera de Python
FROM python:3.9-slim

# Establecemos el directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos todos los archivos del proyecto al contenedor
COPY . /app

# Instalamos las dependencias especificadas en requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Creamos la carpeta necesaria para los modelos
RUN mkdir -p models

# Ejecutamos el entrenamiento de los modelos (genera los .pkl)
RUN python pipelines/train_models.py

# Exponemos el puerto para la aplicación
EXPOSE 8000

# Comando para iniciar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
