# Utiliser une image de base avec support CUDA si possible
FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système pour OpenCV/Pillow si nécessaire
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/train.py"]
