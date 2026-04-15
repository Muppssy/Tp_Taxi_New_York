import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import common

# --- Charger les données d'entraînement ---
def load_train_data():
    print(f"Lecture des données train depuis : {common.DB_PATH}")
    with sqlite3.connect(common.DB_PATH) as con:
        data = pd.read_sql('SELECT * FROM train', con)
    X = data.drop(columns=['trip_duration'])
    y = data['trip_duration']
    return X, y

# --- Charger les données de test ---
def load_test_data():
    print(f"Lecture des données test depuis : {common.DB_PATH}")
    with sqlite3.connect(common.DB_PATH) as con:
        data = pd.read_sql('SELECT * FROM test', con)
    X = data.drop(columns=['trip_duration'])
    y = data['trip_duration']
    return X, y