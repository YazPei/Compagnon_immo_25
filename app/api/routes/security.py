import os

from fastapi import Header, HTTPException

# Par défaut, aligné avec les tests
API_KEY_EXPECTED = os.environ.get("API_KEY", "secret-test-key")


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> str:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API Key missing")
    if x_api_key != API_KEY_EXPECTED:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key
