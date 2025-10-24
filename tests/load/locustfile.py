from locust import HttpUser, task, between
import json
import random

class ImmoAPIUser(HttpUser):
    """Utilisateur simulé pour les tests de charge."""
    
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialisation de l'utilisateur."""
        # Optionnel: authentification
        pass
    
    @task(3)
    def get_metrics(self):
        """Test de l'endpoint metrics."""
        self.client.get("/metrics")
    
    @task(5)
    def predict_price(self):
        """Test de prédiction de prix."""
        sample_data = {
            "surface": random.randint(30, 200),
            "pieces": random.randint(1, 6),
            "ville": random.choice(["Paris", "Lyon", "Marseille"]),
            "type": random.choice(["appartement", "maison"])
        }
        
        with self.client.post(
            "/api/v1/predict",
            json=sample_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data:
                    response.success()
                else:
                    response.failure("No prediction in response")
            else:
                response.failure(f"Status code: {response.status_code}")
    
    @task(2)
    def get_model_info(self):
        """Test d'information sur le modèle."""
        self.client.get("/api/v1/model/info")
    
    @task(1)
    def health_check(self):
        """Test de santé de l'API."""
        self.client.get("/health")

class AdminUser(HttpUser):
    """Utilisateur admin pour tester les endpoints sensibles."""
    
    wait_time = between(5, 10)
    weight = 1  # Moins d'utilisateurs admin
    
    @task
    def admin_dashboard(self):
        """Test du dashboard admin."""
        self.client.get("/admin/dashboard")
    
    @task
    def model_management(self):
        """Test de gestion des modèles."""
        self.client.get("/admin/models")

class DataUploadUser(HttpUser):
    """Utilisateur pour tester l'upload de données."""
    
    wait_time = between(10, 30)
    weight = 1
    
    @task
    def upload_data(self):
        """Test d'upload de données."""
        # Simuler un fichier CSV
        files = {
            'file': ('test_data.csv', 'prix,surface,pieces\n150000,50,2\n', 'text/csv')
        }
        
        with self.client.post(
            "/api/v1/data/upload",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")
