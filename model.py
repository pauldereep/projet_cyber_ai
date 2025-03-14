import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import seaborn as sns
import joblib


# Fixer une graine aléatoire pour la reproductibilité
RANDOM_STATE_SEED = 12

# Charger les dataset
dataset = pd.read_csv("hf://datasets/pauldereep/projet_cyber_ai/dataset.csv")



# Vérifier si la colonne Timestamp existe et la convertir en format numérique
if "Timestamp" in dataset.columns:
    dataset["Timestamp"] = pd.to_datetime(dataset["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors='coerce')
    dataset["Timestamp"] = dataset["Timestamp"].astype(int) / 10**9  # Convertir en secondes

# Séparation des données en train (80%) et test (20%)
train, test = train_test_split(dataset, test_size=0.3, random_state=RANDOM_STATE_SEED)

# Remplacement des valeurs infinies par NaN
train.replace([np.inf, -np.inf], np.nan, inplace=True)
test.replace([np.inf, -np.inf], np.nan, inplace=True)

train.dropna(inplace=True)
test.dropna(inplace=True)

# Sélection des colonnes numériques pour le scaling
numerical_columns = [col for col in dataset.columns if col not in ["Label"]]

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


# Validation croisée avec 5 folds
cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5)

print(f"\n Scores de validation croisée : {cv_scores}")
print(f" Moyenne des scores de validation croisée : {cv_scores.mean():.4f}")

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

# Sauvegarde du modèle et du scaler
joblib.dump(log_reg, 'logistic_regression_model.pkl')


