import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response, UploadFile

from policyiq.db import db_healthy
from policyiq.ingest.pipeline import ingest_pdf
from policyiq.llm import get_provider
from policyiq.schemas import IngestResult


def llm_healthy() -> bool:
    """Wrapped as a module-level function so readiness has one seam to check and to
    substitute in tests, mirroring db_healthy."""
    return get_provider().healthy()

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
    llm = "ok" if llm_healthy() else "unavailable"

    # Only the database gates readiness. The model server is a single external
    # dependency shared by every replica, so marking them all unready during its outage
    # removes the service from the load balancer without bringing it back - and stops
    # ingestion, which needs no model at all, turning a partial outage into a total one.
    # Nothing works without the database, so that one does gate.
    #
    # The model server's state is still reported, because an operator needs to see it.
    if database != "ok":
        response.status_code = 503
    return {"database": database, "llm": llm}


@app.post("/v1/ingest", response_model=IngestResult)
def ingest(file: UploadFile) -> IngestResult:
    """Accept a policy PDF, store its chunks, and report what was stored.

    Defined with `def` rather than `async def` on purpose. Parsing, embedding and the
    database write all block; on the event loop they would stall every other request
    for the length of an ingest. FastAPI runs a synchronous route in a worker thread,
    which is the correct place for blocking work.
    """
    # The uploaded name is client-supplied and goes on to be used as a path. Taking
    # only the final component prevents a name like "../../etc/cron.d/x.pdf" from
    # writing outside the temporary directory, and keeps the stored document identity
    # to a plain filename.
    safe_name = Path(file.filename or "").name
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="expected a .pdf file")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / safe_name
        with path.open("wb") as out:
            # Streamed rather than read into memory, so a large policy does not cost
            # its own size in RAM.
            shutil.copyfileobj(file.file, out)
        try:
            return ingest_pdf(path)
        except ValueError as exc:
            # A document we cannot read is a problem with the request, not the server.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
