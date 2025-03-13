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

# Charger le dataset
df_dataset = pd.read_csv("./dataset/02-14-2018.csv")

# Supprimer les valeurs manquantes initiales
df_dataset.dropna(inplace=True)


# Équilibrer le dataset (même nombre d'exemples pour chaque classe)
df1 = df_dataset[df_dataset["Label"] == "Benign"][:761886]
df2 = df_dataset[(df_dataset["Label"] == "FTP-BruteForce")][:380943]
df3 = df_dataset[df_dataset["Label"] == "SSH-Bruteforce"][:380943]
df_equal = pd.concat([df1, df2, df3], axis=0)
df_equal.replace(to_replace="Benign", value=0, inplace=True)
df_equal.replace(to_replace="SSH-Bruteforce", value=1, inplace=True)
df_equal.replace(to_replace="FTP-BruteForce", value=2, inplace=True)

# Vérifier si la colonne Timestamp existe et la convertir en format numérique
if "Timestamp" in df_equal.columns:
    df_equal["Timestamp"] = pd.to_datetime(df_equal["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    df_equal["Timestamp"] = df_equal["Timestamp"].astype(int) / 10**9  # Convertir en secondes

# Séparation des données en train (80%) et test (20%)
train, test = train_test_split(df_equal, test_size=0.3, random_state=RANDOM_STATE_SEED)

# Remplacement des valeurs infinies par NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Sélection des colonnes numériques pour le scaling
numerical_columns = [col for col in df_equal.columns if col not in ["Label"]]

# Appliquer la normalisation Min-Max
scaler = MinMaxScaler().fit(train[numerical_columns])
train[numerical_columns] = scaler.transform(train[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])

# Séparation des features et du label
y_train = np.array(train.pop("Label"))
X_train = train.values
y_test = np.array(test.pop("Label"))
X_test = test.values

# Gérer les NaN en remplaçant par 0
X_train = pd.DataFrame(X_train).fillna(0).values
X_test = pd.DataFrame(X_test).fillna(0).values

# Initialisation du modèle de régression logistique
log_reg = LogisticRegression(random_state=RANDOM_STATE_SEED, max_iter=1000)

# Entraînement du modèle
log_reg.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "SSH-Bruteforce", "FTP-BruteForce"], 
            yticklabels=["Benign", "SSH-Bruteforce", "FTP-BruteForce"])
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.title("Matrice de confusion")
plt.show()
