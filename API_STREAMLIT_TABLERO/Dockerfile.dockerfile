FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar todos los archivos al contenedor
COPY . /app

# Instalar dependencias
RUN pip install --upgrade pip && pip install -r requirements.txt

# copiamos archivos necesarios
COPY data /app/data
COPY models /app/models

# Crear carpetas necesarias
RUN mkdir -p data models

# Ejecutar preprocesamiento y entrenamiento
RUN python pipelines/data_processing.py
RUN python pipelines/train_models.py

# Exponer el puerto para Railway
EXPOSE 8000

# Comando para iniciar Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
