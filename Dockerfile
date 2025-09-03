# Utilisez un multi-stage build pour réduire la taille de l'image
FROM python:3.9-slim as builder
WORKDIR /app
COPY pyproject.toml pdm.lock ./
RUN pip install pdm && \
    pdm export -f requirements --prod > requirements.txt && \
    pip install --prefix=/install -r requirements.txt

FROM python:3.9-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ /app/src/
COPY models/ /app/models/

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["uvicorn", "src.serving.model_serving:app", "--host", "0.0.0.0", "--port", "8000"]
