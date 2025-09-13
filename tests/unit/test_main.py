"""
Test the main FastAPI application.

This file tests that our basic FastAPI app works and the health endpoints respond correctly.
"""

from fastapi.testclient import TestClient
from muzzle.main import app

# Create a test client for our FastAPI app
client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint returns basic app info."""
    response = client.get("/")

    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200
    data = response.json()

    assert data["name"] == "Muzzle"
    assert data["status"] == "healthy"
    assert "version" in data


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/health")

    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "Muzzle"
    assert "version" in data