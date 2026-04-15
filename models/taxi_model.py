import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

# Classe qui emballe le modèle + tout son preprocessing
class TaxiModel:
    def __init__(self, model):
        self.model = model  # le modèle Ridge à l'intérieur

    def __preprocess(self, X):
        X = X.copy()
        X = X.drop(columns=['id', 'dropoff_datetime'])
        X['pickup_datetime'] = pd.to_datetime(X['pickup_datetime'])
        X['weekday'] = X['pickup_datetime'].dt.weekday
        X['month']   = X['pickup_datetime'].dt.month
        X['hour']    = X['pickup_datetime'].dt.hour
        lat1 = np.radians(X['pickup_latitude'])
        lat2 = np.radians(X['dropoff_latitude'])
        lng1 = np.radians(X['pickup_longitude'])
        lng2 = np.radians(X['dropoff_longitude'])
        dlat, dlng = lat2 - lat1, lng2 - lng1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2)**2
        X['log_distance'] = np.log1p(2 * 6371 * np.arcsin(np.sqrt(a)))
        return X[['log_distance', 'hour', 'weekday', 'month']]

    def predict(self, X):
        # preprocessing + prédiction + reconversion en secondes
        X_processed = self.__preprocess(X)
        y_log = self.model.predict(X_processed)
        return int(np.expm1(y_log[0]))  # retourne directement en secondes