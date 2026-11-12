# SageMaker-compatible inference image (GET /ping, POST /invocations on :8080).
# Build from the repository root:   docker build -f serving/Dockerfile -t shift-aware-rul .
# Run locally with a bundle:        docker run -p 8080:8080 -v $PWD/model:/opt/ml/model:ro shift-aware-rul serve
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /opt/program

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]" \
    && useradd --create-home --uid 10001 model

USER model
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/ping')"
ENTRYPOINT ["python", "-m", "rul.serving"]
CMD ["serve"]
