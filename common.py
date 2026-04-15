# Fichier central pour stocker les chemins du projet et les constantes communes
from pathlib import Path

# Dossier racine du projet (là où se trouve ce fichier)
BASE_DIR = Path(__file__).resolve().parent

# Dossiers principaux
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebook"

# Fichier de la base de données SQLite (créé automatiquement par download_data.py)
DB_PATH = DATA_DIR / "taxi.db"

# Fichier du modèle entraîné (créé automatiquement par train.py)
MODEL_PATH = MODELS_DIR / "model.pkl"

# Constante pour la reproductibilité (même valeur que dans le notebook)
RANDOM_STATE = 42