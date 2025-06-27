import pickle
import re

# Chemin du fichier à nettoyer
PTH = 'api_test/models/features_40.pkl'

with open(PTH, 'rb') as f:
    features = pickle.load(f)

features_clean = [f for f in features if isinstance(f, str) and not re.fullmatch(r'\d+', f)]

print(f"Features conservées ({len(features_clean)}) :\n", features_clean)

with open(PTH, 'wb') as f:
    pickle.dump(features_clean, f)

print("Fichier features_40.pkl nettoyé et sauvegardé.") 