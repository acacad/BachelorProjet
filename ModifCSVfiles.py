import pandas as pd
from datetime import datetime, timedelta

# Chemin direct vers le fichier CSV (à adapter selon ton fichier réel)
csv_path = "C:/Users/Andresin/Downloads/marketing_sales_data.csv"

# Lecture du CSV
df = pd.read_csv(csv_path)

# Supposons que tu veux ajouter une colonne de dates à partir d'aujourd'hui vers le passé
# Exemple : dernière ligne = aujourd'hui, ligne précédente = aujourd'hui -1j, etc.
aujourd_hui = datetime.today().date()
nb_lignes = len(df)
dates = [aujourd_hui - timedelta(days=i) for i in reversed(range(nb_lignes))]

# Ajout de la colonne 'date'
df["date"] = dates

# Affichage pour vérification
print(df.head())

# Sauvegarde éventuelle (optionnelle)
df.to_csv("C:/Users/Andresin/Downloads/marketing_sales_date_data.csv", index=False)
