"""
Locust Load Testing & Latency Benchmarking Suite for ChurnGuard API.
Simulates concurrent user behavior across single prediction, batch prediction, and monitoring endpoints.

Usage:
  locust -f scripts/locustfile.py --headless -u 50 -r 10 --run-time 1m --host http://localhost:8000
"""

import os
import random
from locust import HttpUser, task, between


class ChurnGuardUser(HttpUser):
    wait_time = between(0.1, 0.5)  # 100ms to 500ms pacing between requests

    def on_start(self):
        self.api_key = os.getenv("API_KEY", "dev-key-change-in-prod")
        self.headers = {"X-API-Key": self.api_key, "Content-Type": "application/json"}
        self.domains = ["telecom", "school", "ecommerce", "fitness"]

    @task(6)
    def predict_single(self):
        """Task weight 60%: Test single customer churn prediction latency."""
        domain = random.choice(self.domains)
        payload = {
            "customerID": f"LOCUST-{random.randint(1000, 9999)}",
            "tenure": random.randint(1, 72),
            "MonthlyCharges": round(random.uniform(20.0, 120.0), 2),
            "TotalCharges": round(random.uniform(100.0, 5000.0), 2),
            "SeniorCitizen": random.choice([0, 1]),
            "gender": random.choice(["Male", "Female"]),
            "Partner": random.choice(["Yes", "No"]),
            "Dependents": random.choice(["Yes", "No"]),
            "PhoneService": "Yes",
            "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
            "InternetService": random.choice(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": random.choice(["Yes", "No", "No internet service"]),
            "OnlineBackup": random.choice(["Yes", "No", "No internet service"]),
            "DeviceProtection": random.choice(["Yes", "No", "No internet service"]),
            "TechSupport": random.choice(["Yes", "No", "No internet service"]),
            "StreamingTV": random.choice(["Yes", "No", "No internet service"]),
            "StreamingMovies": random.choice(["Yes", "No", "No internet service"]),
            "Contract": random.choice(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": random.choice(["Yes", "No"]),
            "PaymentMethod": random.choice([
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ]),
        }
        self.client.post(
            f"/predict?domain={domain}",
            json=payload,
            headers=self.headers,
            name="/predict (Single)",
        )

    @task(25)
    def predict_batch(self):
        """Task weight 25%: Test batch prediction throughput and latency."""
        batch_size = random.randint(5, 20)
        customers = []
        for i in range(batch_size):
            customers.append({
                "customerID": f"BATCH-{i}",
                "tenure": random.randint(1, 60),
                "MonthlyCharges": round(random.uniform(30.0, 100.0), 2),
                "TotalCharges": round(random.uniform(100.0, 3000.0), 2),
                "SeniorCitizen": 0,
                "gender": "Female",
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
            })
        self.client.post(
            "/predict/batch",
            json={"customers": customers},
            headers=self.headers,
            name="/predict/batch",
        )

    @task(1)
    def health_check(self):
        """Task weight 10%: Health check endpoint."""
        self.client.get("/health", name="/health")

    @task(1)
    def drift_status(self):
        """Task weight 5%: Drift monitoring status."""
        self.client.get("/drift/status", name="/drift/status")
