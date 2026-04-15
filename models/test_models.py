import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd
import sqlite3
import common
from models.train import preprocess

# --- Charger le modèle depuis le fichier ---
def load_model():
    print(f"Chargement du modèle : {common.MODEL_PATH}")
    with open(common.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

# --- Prendre 5 lignes aléatoires depuis la base ---
def load_sample():
    with sqlite3.connect(common.DB_PATH) as con:
        data = pd.read_sql('SELECT * FROM test ORDER BY RANDOM() LIMIT 5', con)
    X = data.drop(columns=['trip_duration'])
    y = data['trip_duration']
    return X, y

if __name__ == "__main__":
    model = load_model()
    X, y = load_sample()

    # Appliquer le même preprocessing que pour l'entraînement
    X_processed, y_log = preprocess(X, y)

    # Prédire (résultat en log → on remet en secondes avec expm1)
    y_pred_log = model.predict(X_processed)
    y_pred_secondes = np.expm1(y_pred_log)

    # Afficher les résultats
    resultats = pd.DataFrame({
        'durée réelle (s)' : y.values,
        'durée prédite (s)': y_pred_secondes.round(0)
    })
    print(resultats)