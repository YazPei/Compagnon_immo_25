from functools import lru_cache
import joblib
import sys
from api_test.app.utils.custom_functions import replace_minus1, plus1, invert11, cyclical_encode

# Ajoute les fonctions dans le scope global pour joblib
sys.modules['__main__'].replace_minus1 = replace_minus1
sys.modules['__main__'].plus1 = plus1
sys.modules['__main__'].invert11 = invert11
sys.modules['__main__'].cyclical_encode = cyclical_encode

@lru_cache(maxsize=1)
def get_model():
    return joblib.load("api_test/models/lightgbm_model.joblib")

@lru_cache(maxsize=1)
def get_preprocessor():
    return joblib.load("api_test/models/preprocessor.joblib")

@lru_cache(maxsize=1)
def get_model_metadata():
    return joblib.load("api_test/models/model_metadata.joblib") 