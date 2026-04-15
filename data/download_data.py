import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
import common

# --- Télécharger les données depuis GitHub ---
def download_data():
    url = "https://github.com/eishkina-estia/ML2023/raw/main/data/New_York_City_Taxi_Trip_Duration.zip"
    print(f"Téléchargement des données depuis : {url}")
    data = pd.read_csv(url, compression='zip')
    print(f"Données téléchargées : {data.shape[0]} lignes, {data.shape[1]} colonnes")
    return data

# --- Sauvegarder dans la base SQLite ---
def save_data(data):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=common.RANDOM_STATE)
    print(f"Sauvegarde dans : {common.DB_PATH}")
    with sqlite3.connect(common.DB_PATH) as con:
        data_train.to_sql(name='train', con=con, if_exists='replace', index=False)
        data_test.to_sql(name='test',  con=con, if_exists='replace', index=False)
    print(f"Train : {len(data_train)} lignes  |  Test : {len(data_test)} lignes")

if __name__ == "__main__":
    data = download_data()
    save_data(data)