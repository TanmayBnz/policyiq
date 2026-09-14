from fastapi.testclient import TestClient

from policyiq.main import app

client = TestClient(app)


def test_healthz_returns_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_reports_dependency_status():
    response = client.get("/readyz")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "database" in body
    assert "llm" in body
