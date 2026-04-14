# Fichier central pour stocker les chemins du projet et les constantes communesfrom pathlib import Path
from pathlib import Path
# dossier racine du projet
BASE_DIR = Path(__file__).resolve().parent

# dossiers principaux
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOK_DIR = BASE_DIR / "notebook"