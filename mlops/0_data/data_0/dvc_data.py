# mlops/0_data/data_0/dvc_data.py
import os
from dagshub.upload import Repo

repo = Repo(owner="YazPei", name="Compagnon_immo", branch="main")

repo.upload(
    local_path="data/merged_sales_data.csv",       # fichier local à envoyer
    remote_path="data/merged_sales_data.csv",      # chemin où il apparaîtra dans le repo
    versioning="git",                              # on pousse en Git (pas DVC)
    commit_message="Add merged_sales_data.csv (API)"
)

print("✅ Upload terminé vers la branche Phase2_test_1")

