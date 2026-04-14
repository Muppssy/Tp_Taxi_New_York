import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from common import DATA_DIR


def find_data_files():
    print(f"Dossier data : {DATA_DIR}")

    if not DATA_DIR.exists():
        print("Le dossier data n'existe pas.")
        return []

    all_files = list(DATA_DIR.iterdir())

    data_files = []
    allowed_extensions = [".csv", ".zip", ".xlsx"]

    for file in all_files:
        if file.is_file() and file.suffix.lower() in allowed_extensions:
            data_files.append(file)

    if not data_files:
        print("Aucun fichier de données trouvé dans le dossier data.")
    else:
        print("Fichiers de données trouvés :")
        for file in data_files:
            print("-", file.name)

    return data_files


if __name__ == "__main__":
    find_data_files()