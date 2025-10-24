from locust import HttpUser, task, between
import json


class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(2)
    def liveness_check(self):
        self.client.get("/liveness")

    @task(2)
    def readiness_check(self):
        self.client.get("/readiness")

    @task(1)
    def root_endpoint(self):
        self.client.get("/")

    @task(5)
    def estimation_request(self):
        # Simulation d'une requête d'estimation immobilière
        payload = {
            "property_type": "apartment",
            "area": 75,
            "rooms": 3,
            "zip_code": "75001",
            "latitude": 48.8566,
            "longitude": 2.3522,
            "year_built": 1990,
            "energy_class": "C",
            "floor": 2,
            "total_floors": 5,
            "has_elevator": True,
            "has_balcony": True,
            "has_parking": False,
            "is_furnished": False
        }

        headers = {'Content-Type': 'application/json'}
        self.client.post("/api/v1/estimation/", json=payload)

    @task(1)
    def metrics_endpoint(self):
        self.client.get("/metrics")

    @task(1)
    def invalid_request(self):
        # Requête invalide pour tester les erreurs
        self.client.post("/api/v1/estimation/", json={})

    @task(1)
    def not_found(self):
        # Endpoint inexistant pour tester les 404
        self.client.get("/nonexistent")
