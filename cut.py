import pandas as pd
import random

# Charger le fichier CSV original
df = pd.read_csv('data.csv')

# Vérifier que le fichier a au moins 300 lignes
if len(df) < 300:
    print("Le fichier a moins de 300 lignes.")
else:
    # Sélectionner aléatoirement 300 lignes
    random_rows = df.sample(n=300, random_state=1)

    # Enregistrer les lignes sélectionnées dans un nouveau fichier CSV
    random_rows.to_csv('./nouveau_fichier.csv', index=False)
    print("Le nouveau fichier CSV a été créé avec succès.")
