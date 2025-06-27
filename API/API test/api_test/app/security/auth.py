from fastapi import Header, HTTPException, status, Depends
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "test-key-123")

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key invalide ou manquante"
        )
    return True 

def get_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clé API invalide"
        )
    return x_api_key 