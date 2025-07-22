#!/bin/bash
set -euo pipefail

TEMPLATE=".env.template"
TARGET=".env.yaz"

if [[ -f "$TARGET" ]]; then
    echo "✅ Fichier $TARGET déjà présent."
else
    echo "📄 Copie du template $TEMPLATE vers $TARGET"
    cp "$TEMPLATE" "$TARGET"

    echo "✍️  Merci de compléter le fichier $TARGET avec tes identifiants DagsHub :"
    echo "  - DVC_USER"
    echo "  - DVC_TOKEN"
    echo "  - (facultatif) MLFLOW_TRACKING_URI, ST_SUFFIX"
    echo ""
    echo "🔁 Puis relance la commande suivante :"
    echo "    make run_dvc"
    exit 0
fi
