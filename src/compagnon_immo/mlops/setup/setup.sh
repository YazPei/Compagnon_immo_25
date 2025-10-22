#!/bin/bash

echo "🔐 Configuration du remote DVC avec DagsHub"

# Nom du remote
REMOTE_NAME="origin"

# URL DagsHub (adapter si nécessaire)
REMOTE_URL="https://dagshub.com/YazPei/Compagnon_immo_25.dvc"

# Demande les infos personnelles
read -p "👤 DagsHub username : " DAGSHUB_USER
read -s -p "🔑 DagsHub token (will stay hidden) : " DAGSHUB_TOKEN
echo ""

# Ajoute le remote s'il n'existe pas encore
dvc remote list | grep -q "$REMOTE_NAME"
if [ $? -ne 0 ]; then
    echo "➕ Ajout du remote $REMOTE_NAME"
    dvc remote add $REMOTE_NAME $REMOTE_URL
    dvc remote default $REMOTE_NAME
fi

# Configuration locale
dvc remote modify $REMOTE_NAME --local auth basic
dvc remote modify $REMOTE_NAME --local user "$DAGSHUB_USER"
dvc remote modify $REMOTE_NAME --local password "$DAGSHUB_TOKEN"

echo "✅ Remote $REMOTE_NAME correctement configuré dans .dvc/config.local"

