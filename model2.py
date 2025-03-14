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

# Charger les dataset
df_dataset1 = pd.read_csv("./dataset/clean/02-14-2018.csv")
df_dataset2 = pd.read_csv("./dataset/clean/02-15-2018.csv")
df_dataset3 = pd.read_csv("./dataset/clean/02-16-2018.csv")
##df_dataset4 = pd.read_csv("./dataset/clean/02-20-2018.csv")
df_dataset5 = pd.read_csv("./dataset/clean/02-21-2018.csv")
##df_dataset8 = pd.read_csv("./dataset/clean/02-28-2018.csv")
##df_dataset9 = pd.read_csv("./dataset/clean/03-01-2018.csv")
df_dataset10 = pd.read_csv("./dataset/clean/03-02-2018.csv")

# Fusionner avec l'ancien dataset
df_combined = pd.concat([df_dataset1, df_dataset2, df_dataset3, df_dataset5, df_dataset10], axis=0)
# Équilibrer le dataset (même nombre d'exemples pour chaque classe)
df1 = df_combined[df_combined["Label"] == "Benign"][:10000]
df2 = df_combined[(df_combined["Label"] == "FTP-BruteForce")][:10000]
df3 = df_combined[df_combined["Label"] == "SSH-Bruteforce"][:10000]
df4 = df_combined[df_combined["Label"] == "DoS attacks-GoldenEye"][:10000]
df5 = df_combined[df_combined["Label"] == "DoS attacks-Slowloris"][:10000]
df6 = df_combined[df_combined["Label"] == "DoS attacks-SlowHTTPTest"][:10000]
df7 = df_combined[df_combined["Label"] == "DoS attacks-Hulk"][:10000]
df8 = df_combined[df_combined["Label"] == "DDoS attacks-LOIC-HTTP"][:10000]
df9 = df_combined[df_combined["Label"] == "DDOS attack-HOIC"][:10000]
df10 = df_combined[df_combined["Label"] == "Bot"][:10000]
print(len(df10))
df_equal = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10], axis=0)
# Remplacement des labels par des valeurs numériques
df_equal.replace(to_replace="Benign", value=0, inplace=True)
df_equal.replace(to_replace="SSH-Bruteforce", value=1, inplace=True)
df_equal.replace(to_replace="FTP-BruteForce", value=2, inplace=True)
df_equal.replace(to_replace="DoS attacks-GoldenEye", value=3, inplace=True)
df_equal.replace(to_replace="DoS attacks-Slowloris", value=4, inplace=True)
df_equal.replace(to_replace="DoS attacks-SlowHTTPTest", value=5, inplace=True)
df_equal.replace(to_replace="DoS attacks-Hulk", value=6, inplace=True)
df_equal.replace(to_replace="DDoS attacks-LOIC-HTTP", value=7, inplace=True)
df_equal.replace(to_replace="DDOS attack-HOIC", value=8, inplace=True)
df_equal.replace(to_replace="Infilteration", value=9, inplace=True)
df_equal.replace(to_replace="Bot", value=10, inplace=True)
df_equal.to_csv("dataset_combined.csv")
# Vérifier si la colonne Timestamp existe et la convertir en format numérique
if "Timestamp" in df_equal.columns:
    df_equal["Timestamp"] = pd.to_datetime(df_equal["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    df_equal["Timestamp"] = df_equal["Timestamp"].astype(int) / 10**9  # Convertir en secondes

# Séparation des données en train (80%) et test (20%)
train, test = train_test_split(df_equal, test_size=0.3, random_state=RANDOM_STATE_SEED)

# Remplacement des valeurs infinies par NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

train.dropna(inplace=True)
test.dropna(inplace=True)

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

# Initialisation du modèle de régression logistique multinomiale
log_reg = LogisticRegression(random_state=RANDOM_STATE_SEED, max_iter=1000, multi_class="multinomial", solver="lbfgs")

# Entraînement du modèle
log_reg.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = log_reg.predict(X_test)

# Probabilités associées aux prédictions
y_probs = log_reg.predict_proba(X_test)

# Définition des niveaux de confiance en fonction des probabilités
def niveau_confiance(proba_max):
    if proba_max < 0.6:
        return 1  # Bas
    elif proba_max < 0.85:
        return 2  # Moyen
    else:
        return 3  # Élevé

# Associer chaque prédiction à un niveau de confiance
niveaux_risque = [niveau_confiance(max(p)) for p in y_probs]

# Résumé des résultats avec niveaux de confiance
result_df = pd.DataFrame({
    "Vraie Classe": y_test,
    "Prédiction": y_pred,
    "Proba max": [max(p) for p in y_probs],
    "Niveau de confiance": niveaux_risque
})

# Afficher les 10 premières lignes des résultats
print("\n🔍 Aperçu des résultats de classification :")
print(result_df.head(10))

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"\n Accuracy: {accuracy:.4f}")
print("\n Confusion Matrix:")
print(conf_matrix)
print("\n Classification Report:")
print(class_report)

# Correction de l'affichage des catégories sur la heatmap
category_labels = ["Benign", "SSH-Bruteforce", "FTP-BruteForce", 
                   "DoS attacks-GoldenEye", "DoS attacks-Slowloris", 
                   "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk", 
                   "DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC", "Bot"]

# Affichage de la matrice de confusion
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=category_labels, 
            yticklabels=category_labels)
plt.xlabel("Prédictions")
plt.ylabel("Vraies valeurs")
plt.title("Matrice de confusion")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.show()


# Affichage de la répartition des niveaux de confiance
plt.figure(figsize=(8, 5))
sns.histplot(result_df["Niveau de confiance"], bins=3, kde=False, discrete=True)
plt.xlabel("Niveau de confiance")
plt.ylabel("Nombre de prédictions")
plt.title("Distribution des niveaux de confiance")
plt.xticks([1, 2, 3], labels=["Bas", "Moyen", "Élevé"])
plt.show()


