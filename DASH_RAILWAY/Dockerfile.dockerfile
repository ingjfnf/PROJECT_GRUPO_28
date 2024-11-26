# Usamos una imagen base ligera de Python
FROM python:3.9-slim

# Establecemos el directorio de trabajo en el contenedor
WORKDIR /app

# Copiamos todos los archivos del proyecto al contenedor
COPY . /app

# Instalamos las dependencias especificadas en requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Aseguramos que la carpeta 'models' esté disponible
COPY models /app/models

# Exponemos el puerto que usará la aplicación
EXPOSE 8000

# Comando para iniciar Streamlit con las configuraciones adecuadas
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
