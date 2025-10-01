# 1) Donne lecture à tous les fichiers, et "traverse" aux dossiers
chmod -R a+rX dags

# (au besoin, plus strict/verbeux)
find dags -type d -exec chmod 755 {} \;
find dags -type f -name "*.py" -exec chmod 644 {} \;

# 2) (optionnel) remet l’ownership à ton user
sudo chown -R "$USER":"$USER" dags

# 3) (optionnel mais sain) retire les CRLF dans les DAGs
find dags -type f -name "*.py" -exec sed -i 's/\r$//' {} \;

# 4) Redémarre Airflow pour rescanner
docker compose restart airflow

