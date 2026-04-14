from pathlib import Path

from pathlib import Path
from common import DATA_DIR

def check_data_folder():
    print(f"Dossier data : {DATA_DIR}")

    if not DATA_DIR.exists():
        print("Le dossier data n'existe pas.")
        return

    files = list(DATA_DIR.iterdir())

    if not files:
        print("Le dossier data est vide.")
    else:
        print("Fichiers trouvés dans data :")
        for file in files:
            print("-", file.name)

if __name__ == "__main__":
    check_data_folder()