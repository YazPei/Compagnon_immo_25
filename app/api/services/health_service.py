# app/api/services/health_service.py
async def check_database():
    return {"status": "healthy"}

async def check_ml_service():
    return {"status": "healthy"}

