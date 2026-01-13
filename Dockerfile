# Dockerfile für Factory Part Recognition
# Multi-stage build für optimale Image-Größe

# === Stage 1: Builder ===
FROM python:3.10-slim as builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# === Stage 2: Runtime ===
FROM python:3.10-slim as runtime

WORKDIR /app

# Non-root user für Security
RUN useradd --create-home --shell /bin/bash appuser

# Python packages aus Builder Stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Anwendungscode kopieren
COPY --chown=appuser:appuser main.py .
COPY --chown=appuser:appuser index.html .
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser config/ ./config/
COPY --chown=appuser:appuser models/ ./models/

# Verzeichnisse erstellen
RUN mkdir -p static logs && chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Port freigeben
EXPOSE 8000

# Startbefehl
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
