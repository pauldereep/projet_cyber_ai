import pandas as pd
import numpy as np
import joblib
import os
import subprocess

# Charger le modèle de classification
model = joblib.load("logistic_regression_model.pkl")

# Charger le scaler MinMaxScaler pour normaliser les nouvelles données
scaler = joblib.load("scaler.pkl")

# Charger les logs à analyser (fichier CSV contenant les paquets sous la bonne forme)
logs_df = pd.read_csv("logs_reseau.csv")

# Vérifier si la colonne Timestamp existe et la convertir au bon format
if "Timestamp" in logs_df.columns:
    logs_df["Timestamp"] = pd.to_datetime(logs_df["Timestamp"]).astype(int) / 10**9

# Normaliser les données (sans la colonne "Label" car c'est une prédiction)
numerical_columns = [col for col in logs_df.columns if col != "Label"]
logs_df[numerical_columns] = scaler.transform(logs_df[numerical_columns])

# Faire les prédictions
predictions = model.predict(logs_df.values)

# Probabilités des prédictions
probs = model.predict_proba(logs_df.values)

# Fonction pour exécuter une action en cas de détection d'attaque
def action_si_menace(ip_source, attaque_detectee):
    print(f"⚠️ ATTENTION : Une attaque détectée ({attaque_detectee}) depuis {ip_source} ! ⚠️")

    # 1️⃣ Enregistrer dans un fichier de logs
    with open("log_alertes.txt", "a") as log_file:
        log_file.write(f"{ip_source} - ATTACK: {attaque_detectee}\n")

    # 2️⃣ Optionnel : Bloquer l'IP (Linux uniquement)
    try:
        subprocess.run(["sudo", "iptables", "-A", "INPUT", "-s", ip_source, "-j", "DROP"], check=True)
        print(f"🚫 IP {ip_source} BLOQUÉE via iptables")
    except Exception as e:
        print(f"❌ Impossible de bloquer l'IP {ip_source} : {e}")

    # 3️⃣ Optionnel : Envoyer une alerte par Discord ou Slack
    # (Exemple : Envoyer un message sur un webhook Discord)
    # webhook_url = "https://discord.com/api/webhooks/XXX"
    # requests.post(webhook_url, json={"content": f"⚠️ ATTACK DETECTED: {attaque_detectee} from {ip_source}!"})

# Vérifier chaque ligne et agir si nécessaire
for index, row in logs_df.iterrows():
    prediction = predictions[index]
    proba_max = max(probs[index])

    if prediction != 0:  # 0 = Normal, toute autre valeur = attaque détectée
        ip_source = row["Src IP"] if "Src IP" in row else "IP inconnue"
        type_attaque = prediction  # La classe prédite correspond au type d'attaque

        action_si_menace(ip_source, type_attaque)

print("✅ Analyse terminée.")
