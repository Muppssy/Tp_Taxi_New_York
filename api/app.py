import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import sqlite3
from fastapi import FastAPI
from pydantic import BaseModel
import common


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

    # TaxiModel.predict() fait preprocessing + prédiction + reconversion
    duree_secondes = model.predict(X)

    save_prediction(trip, duree_secondes)
    return {"durée_prédite_secondes": duree_secondes}


# --- Endpoint /predictions : lire les prédictions sauvegardées ---
@app.get("/predictions")
def get_predictions():
    with sqlite3.connect(common.DB_PATH) as con:
        data = pd.read_sql('SELECT * FROM predictions', con)
    return data.to_dict(orient='records')