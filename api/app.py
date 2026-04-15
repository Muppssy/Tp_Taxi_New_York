import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import common
from models.train import preprocess

# --- Charger le modèle au démarrage de l'API ---
with open(common.MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

# --- Schéma des données d'entrée ---
class TripInput(BaseModel):
    pickup_datetime:   str
    pickup_latitude:   float
    pickup_longitude:  float
    dropoff_latitude:  float
    dropoff_longitude: float

# --- Sauvegarder la prédiction dans la base ---
def save_prediction(trip: TripInput, duree_secondes: int):
    with sqlite3.connect(common.DB_PATH) as con:
        pd.DataFrame([{
            'pickup_datetime':   trip.pickup_datetime,
            'pickup_latitude':   trip.pickup_latitude,
            'pickup_longitude':  trip.pickup_longitude,
            'dropoff_latitude':  trip.dropoff_latitude,
            'dropoff_longitude': trip.dropoff_longitude,
            'predicted_duration': duree_secondes
        }]).to_sql(name='predictions', con=con, if_exists='append', index=False)

# --- Endpoint /predict ---
@app.post("/predict")
def predict(trip: TripInput):
    X = pd.DataFrame([{
        'id': 0,
        'dropoff_datetime': '2016-01-01 00:00:00',
        'pickup_datetime':  trip.pickup_datetime,
        'pickup_latitude':  trip.pickup_latitude,
        'pickup_longitude': trip.pickup_longitude,
        'dropoff_latitude': trip.dropoff_latitude,
        'dropoff_longitude':trip.dropoff_longitude,
    }])

    y_dummy = pd.Series([0])
    X_processed, _ = preprocess(X, y_dummy)

    y_pred_log = model.predict(X_processed)
    duree_secondes = int(np.expm1(y_pred_log[0]))

    # Sauvegarder dans taxi.db
    save_prediction(trip, duree_secondes)

    return {"durée_prédite_secondes": duree_secondes}