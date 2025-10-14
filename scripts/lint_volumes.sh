#!/bin/bash
# Vérifie que tous les volumes dans docker-compose.yml utilisent :Z ou :z

set -e

files=$(find . -name "docker-compose*.yml")

fail=0

for file in $files; do
  while read -r line; do
    # Ignore les lignes qui ne sont pas des montages de volumes
    if [[ "$line" =~ ^[[:space:]]*-[[:space:]]*[^:]+:[^:]+ ]]; then
      # Vérifie la présence de :Z ou :z à la fin du volume
      if ! [[ "$line" =~ :[Zz][[:space:]]*$ ]]; then
        echo "Volume sans :Z ou :z détecté dans $file :"
        echo "  $line"
        fail=1
      fi
    fi
  done < "$file"
done

if [ $fail -eq 1 ]; then
  echo "Erreur : certains volumes n'utilisent pas l'option SELinux :Z ou :z."
  exit 1
else
  echo "Tous les volumes utilisent correctement :Z ou :z."
fi
