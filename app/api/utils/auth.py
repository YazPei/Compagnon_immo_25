# app/api/utils/auth.py

from fastapi import Header, HTTPException, status


async def get_api_key(x_api_key: str = Header(default=None)):
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API Key"
        )
    if x_api_key != "test_api_key":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return x_api_key
