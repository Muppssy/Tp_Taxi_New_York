import sys
from pathlib import Path

import numpy as np
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import pandas as pd
import sqlite3
from fastapi import FastAPI
from fastapi import HTTPException
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
    
# --- Validation des coordonnées ---
def validate_trip(trip: TripInput):
    # Vérifier que les coordonnées sont dans les limites globales
    if not (-90 <= trip.pickup_latitude <= 90):
        raise HTTPException(status_code=400, detail="pickup_latitude invalide (-90 à 90)")
    if not (-90 <= trip.dropoff_latitude <= 90):
        raise HTTPException(status_code=400, detail="dropoff_latitude invalide (-90 à 90)")
    if not (-180 <= trip.pickup_longitude <= 180):
        raise HTTPException(status_code=400, detail="pickup_longitude invalide (-180 à 180)")
    if not (-180 <= trip.dropoff_longitude <= 180):
        raise HTTPException(status_code=400, detail="dropoff_longitude invalide (-180 à 180)")

    # Vérifier que le trajet fait plus de 50 mètres
    lat1, lat2 = np.radians(trip.pickup_latitude), np.radians(trip.dropoff_latitude)
    lng1, lng2 = np.radians(trip.pickup_longitude), np.radians(trip.dropoff_longitude)
    dlat, dlng = lat2 - lat1, lng2 - lng1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
    distance_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    if distance_km < 0.05:  # 50 mètres = 0.05 km
        raise HTTPException(status_code=400, detail="Distance trop courte (minimum 50m)")

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
    validate_trip(trip) 
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