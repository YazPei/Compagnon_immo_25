import pickle
import re
import logging
from pathlib import Path
from typing import List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_features(file_path: str) -> List[str]:
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    
    try:
        backup_path = path.with_suffix('.pkl.backup')
        if not backup_path.exists():
            path.rename(backup_path)
            logger.info(f"Backup créé : {backup_path}")
        
        with open(backup_path, 'rb') as f:
            features = pickle.load(f)
        
        features_clean = [
            f for f in features 
            if isinstance(f, str) 
            and f.strip()  
            and not re.fullmatch(r'\d+', f)  
            and len(f) > 1 
        ]
        
        logger.info(f"Features avant nettoyage : {len(features)}")
        logger.info(f"Features après nettoyage : {len(features_clean)}")
        
        with open(path, 'wb') as f:
            pickle.dump(features_clean, f)
        
        return features_clean
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage : {e}")
        raise

if __name__ == "__main__":
    PTH = 'api_test/models/features_40.pkl'
    try:
        features_clean = clean_features(PTH)
        print(f"✅ Nettoyage terminé. {len(features_clean)} features conservées.")
        print("Exemple de features :", features_clean[:5])
    except Exception as e:
        print(f"Erreur : {e}")