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


def test_readiness_reports_the_model_server_separately():
    body = client.get("/readyz").json()

    assert body["database"] in ("ok", "unavailable")
    assert body["llm"] in ("ok", "unavailable")


def test_the_model_server_being_down_does_not_make_the_service_unready(monkeypatch):
    """Readiness governs whether this instance receives traffic at all.

    Ollama is a single external dependency shared by every replica, so marking all of
    them unready during an outage removes the service from the load balancer without
    bringing the model server back - and takes ingestion, which does not need a model
    at all, down with it. A partial outage would become a total one. The database is
    different: nothing works without it.
    """
    import policyiq.main as main

    monkeypatch.setattr(main, "llm_healthy", lambda: False)
    monkeypatch.setattr(main, "db_healthy", lambda: True)

    response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"database": "ok", "llm": "unavailable"}


def test_the_database_being_down_does_make_the_service_unready(monkeypatch):
    import policyiq.main as main

    monkeypatch.setattr(main, "db_healthy", lambda: False)
    monkeypatch.setattr(main, "llm_healthy", lambda: True)

    assert client.get("/readyz").status_code == 503
