import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Fixer une graine aléatoire pour la reproductibilité
RANDOM_STATE_SEED = 12


def save_modified_csv(dataset_equal):

    # Enregistrer le DataFrame modifié dans un nouveau fichier CSV
    dataset_equal.to_csv('./new_data/new_03-01-2018.csv', index=False)
    print("Le DataFrame modifié a été enregistré dans 'nouveau_fichier.csv'.")

def count_labels(dataset):
    nom_colonne = 'Label'
    valeur_label_1 = 'Infilteration'
    valeur_label_2 = 'Other'

    # Filtrer les lignes contenant le label spécifique
    lignes_avec_label_1 = dataset[dataset[nom_colonne] == valeur_label_1]
    lignes_avec_label_2 = dataset[dataset[nom_colonne] == valeur_label_2]
    # Compter le nombre de lignes
    nombre_de_lignes = len(lignes_avec_label_1)
    print(f"Il y a {nombre_de_lignes} lignes avec le label '{valeur_label_1}'.")

    nombre_de_lignes = len(lignes_avec_label_2)
    print(f"Il y a {nombre_de_lignes} lignes avec le label '{valeur_label_2}'.")


# Charger le dataset
dataset = pd.read_csv("./data/03-01-2018.csv", low_memory=False)


# Supprimer les valeurs manquantes et infinies
dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)


if "Timestamp" in dataset.columns:
    dataset["Timestamp"] = pd.to_datetime(dataset["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    dataset["Timestamp"] = dataset["Timestamp"].astype(int) / 10**9  # Convertir en secondes


# Équilibrer le dataset (même nombre d'exemples pour chaque classe)
d1 = dataset[dataset["Label"] == "Benign"][:10000]
d2 = dataset[(dataset["Label"] == "Infilteration")][:10000]

dataset_equal = pd.concat([d1, d2], axis=0)
dataset_equal.replace(to_replace="Benign", value=0, inplace=True)
dataset_equal.replace(to_replace="Infilteration", value=1, inplace=True)


# Séparation des données en train (80%) et test (30%)
train, test = train_test_split(dataset_equal, test_size=0.3, random_state=RANDOM_STATE_SEED)

# Supprimer les valeurs manquantes et infinies
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train.dropna(inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)
test.dropna(inplace=True)


# Sélection des colonnes numériques pour le scaling
numerical_columns = [col for col in dataset_equal.columns if col not in ["Label"]]

# Appliquer la normalisation Min-Max
scaler = MinMaxScaler().fit(train[numerical_columns])
train[numerical_columns] = scaler.transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])

# Séparation des features et du label
y_train = np.array(train.pop("Label"))
x_train = train.values
y_test = np.array(test.pop("Label"))
x_test = test.values

#Initialiser le modèle de régression logistique
model = LogisticRegression(random_state=RANDOM_STATE_SEED, max_iter=10000)

# Entraîner le modèle
model.fit(x_train, y_train)

# Prédiction sur l'ensemble du test
y_pred = model.predict(x_test)

#Evaluation du modèle
classification_report = classification_report(y_test, y_pred)

# Afficher les résultats
print(f"\nClassification Report:\n{classification_report}")

# Matrice de confusion
confusion_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()




#save_modified_csv(dataset_equal)





    

