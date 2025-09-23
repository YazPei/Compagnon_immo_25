from typing import Dict


# app/api/services/health_service.py
async def check_database() -> Dict[str, str]:
    return {"status": "healthy"}


async def check_ml_service() -> Dict[str, str]:
    return {"status": "healthy"}

