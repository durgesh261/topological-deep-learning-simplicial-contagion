import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Topological Deep Learning" in data["service"]
    assert "simplices" in data


def test_topology_metrics_endpoint(client):
    response = client.get("/api/v1/topology")
    assert response.status_code == 200
    data = response.json()
    assert data["fundamental_theorem_holds"] is True
    assert data["betti_0"] >= 1
    assert data["betti_1"] >= 0


def test_hodge_decomposition_endpoint(client):
    response = client.get("/api/v1/hodge-decomposition")
    assert response.status_code == 200
    data = response.json()
    assert "gradient_flow_pct" in data
    assert "solenoidal_curl_flow_pct" in data
    assert "harmonic_cavity_flow_pct" in data


def test_simulate_cascade_endpoint(client):
    payload = {
        "shock_bank_id": 0,
        "shock_amount_meur": 25000.0
    }
    response = client.post("/api/v1/simulate-cascade", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["shocked_bank"] == 0
    assert "defaulted_banks_count" in data
    assert data["scnn_prediction_accuracy_pct"] >= 0.0
