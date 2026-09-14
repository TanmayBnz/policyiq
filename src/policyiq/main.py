from fastapi import FastAPI, Response

from policyiq.db import db_healthy

app = FastAPI(title="PolicyIQ", version="0.1.0")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is running. Never touches dependencies.

    If this checked Postgres, a database outage would make Kubernetes restart a
    perfectly healthy container — repeatedly, and without ever fixing anything.
    """
    return {"status": "ok"}


@app.get("/readyz")
def readyz(response: Response) -> dict[str, str]:
    """Readiness: dependencies are reachable. Kubernetes uses this to decide whether
    to route traffic here. A failing readiness check removes the pod from the load
    balancer instead of killing it."""
    database = "ok" if db_healthy() else "unavailable"
    llm = "unknown"  # wired in Task 8
    if database != "ok":
        response.status_code = 503
    return {"database": database, "llm": llm}
