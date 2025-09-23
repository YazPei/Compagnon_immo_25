from fastapi.testclient import TestClient
from typing import List, Dict, Any, cast

from app.api.main import app

client = TestClient(app)


class TestBaseEndpoints:
    """Tests des endpoints de base de l'application réelle."""

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data
        assert data.get("status") == "running"

    def test_health_endpoint(self):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ["ok", "healthy"]

    def test_openapi_schema(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema: Dict[str, Any] = response.json()
        paths: Dict[str, Any] = cast(Dict[str, Any], schema.get("paths", {}))
        operation_ids: List[str] = []
        for path_item in paths.values():
            methods: Dict[str, Any] = cast(Dict[str, Any], path_item)
            for method_details in methods.values():
                details: Dict[str, Any] = cast(Dict[str, Any], method_details)
                op_id = details.get("operationId")
                if isinstance(op_id, str) and op_id:
                    operation_ids.append(op_id)
        assert len(operation_ids) == len(set(operation_ids))


        