import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.taxi_model import TaxiModel
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import common
from data.load_data import load_train_data, load_test_data

# --- ÉTAPE 1 : Preprocessing ---
# Transforme les données brutes en données utilisables par le modèle
def preprocess(X, y):
    X = X.copy()

    # Supprimer les colonnes inutiles
    X = X.drop(columns=['id', 'dropoff_datetime'])

    # Convertir la date en vrai type datetime
    X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])

    # Créer de nouvelles colonnes à partir de la date
    X['weekday'] = X['pickup_datetime'].dt.weekday  # lundi=0, dimanche=6
    X['month']   = X['pickup_datetime'].dt.month
    X['hour']    = X['pickup_datetime'].dt.hour

    # Calculer la distance entre départ et arrivée (en km)
    lat1 = np.radians(X['pickup_latitude'])
    lat2 = np.radians(X['dropoff_latitude'])
    lng1 = np.radians(X['pickup_longitude'])
    lng2 = np.radians(X['dropoff_longitude'])
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    X['log_distance'] = np.log1p(distance_km)  # log pour linéariser

    # Transformer la cible (durée en secondes → log)
    y = np.log1p(y)

    # Garder uniquement les colonnes utiles pour le modèle
    features = ['log_distance', 'hour', 'weekday', 'month']
    return X[features], y

# --- ÉTAPE 2 : Entraînement ---
def train_model():
    X_train, y_train = load_train_data()
    X_test,  y_test  = load_test_data()

    X_train, y_train = preprocess(X_train, y_train)
    X_test,  y_test  = preprocess(X_test,  y_test)

    model = Ridge()
    model.fit(X_train, y_train)

    score_train = root_mean_squared_error(y_train, model.predict(X_train))
    score_test  = root_mean_squared_error(y_test,  model.predict(X_test))
    print(f"RMSLE train : {score_train:.4f}")
    print(f"RMSLE test  : {score_test:.4f}")
    
    model = TaxiModel(model)

    return model

# --- ÉTAPE 3 : Sauvegarder le modèle ---
def save_model(model):
    common.MODELS_DIR.mkdir(exist_ok=True)
    with open(common.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modèle sauvegardé : {common.MODEL_PATH}")

# --- Point d'entrée ---
if __name__ == "__main__":
    model = train_model()
    save_model(model)