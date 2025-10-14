#!/bin/bash

set -euo pipefail

echo "⚠️ Réinitialisation complète du dépôt DVC local..."

# === 🧼 Suppression des traces DVC locales ===
if [ -d ".dvc" ]; then
    echo "🧹 Suppression du dossier .dvc (force root safe)"
    rm -rf .dvc || {
        echo "❌ Impossible de supprimer .dvc. Essayez avec : sudo rm -rf .dvc"
        exit 1
    }
fi

rm -f dvc.lock dvc.yaml
find . -name "*.dvc" -delete

# === 📦 Chargement des variables d’environnement ===
ENV_FILE=".env.yaz"
if [ -f "$ENV_FILE" ]; then
    echo "📦 Chargement des variables depuis $ENV_FILE"
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
else
    echo "❌ Fichier $ENV_FILE introuvable. Abandon."
    exit 1
fi

# === ⚙️ Réinitialisation du dépôt DVC ===
echo "⚙️ Initialisation d’un nouveau dépôt DVC..."
dvc init --quiet

# === 🔐 Configuration du remote vers DagsHub ===
echo "🔗 Configuration du remote DVC vers DagsHub..."
dvc remote add -d origin "https://dagshub.com/${DVC_USER}/compagnon_immo_25.dvc.git"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DVC_USER"
dvc remote modify origin --local password "$DVC_TOKEN"

# === 📥 Pull DVC ===
echo "📥 Pull des données trackées depuis DagsHub..."
dvc pull --force

echo "✅ Dépôt DVC restauré depuis DagsHub avec succès."

