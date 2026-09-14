# Minimal and working. Hardened (multi-stage, non-root, pinned digest) on day 10.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir uv && uv pip install --system -e .
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "policyiq.main:app", "--host", "0.0.0.0", "--port", "8000"]
