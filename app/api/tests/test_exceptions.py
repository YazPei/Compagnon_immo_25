import json
import pytest
from starlette.requests import Request
from fastapi.exceptions import RequestValidationError

from app.api.main import (
    validation_exception_handler,
    general_exception_handler,
)


@pytest.mark.asyncio
async def test_validation_exception_handler():
    """Test du gestionnaire d'exceptions de validation."""
    request = Request(scope={"type": "http"})
    exception = RequestValidationError([
        {"loc": ["body"], "msg": "Invalid input"}
    ])

    response = await validation_exception_handler(request, exception)
    assert response.status_code == 422
    # Utiliser json.loads pour comparer les objets JSON sans tenir compte du formatage
    assert json.loads(response.body.decode()) == {"detail": "Validation Error"}


@pytest.mark.asyncio
async def test_general_exception_handler():
    """Test du gestionnaire d'exceptions générales."""
    request = Request(scope={"type": "http"})
    exception = Exception("Erreur interne")

    response = await general_exception_handler(request, exception)
    assert response.status_code == 500
    # Utiliser json.loads pour comparer les objets JSON sans tenir compte du formatage
    assert json.loads(response.body.decode()) == {"detail": "Internal Server Error"}
