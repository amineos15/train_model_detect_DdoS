# Utiliser une image de base officielle de Python avec support de TensorFlow
FROM tensorflow/tensorflow:2.4.1

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers requirements.txt et train_model.py dans le répertoire de travail du conteneur
COPY requirements.txt .
COPY train_model.py .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exécuter le script Python lorsque le conteneur démarre
CMD ["python", "train_model.py"]
