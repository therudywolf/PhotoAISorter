FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

RUN useradd --create-home --no-log-init appuser \
    && chown appuser:appuser /app

COPY --chown=appuser:appuser . .

RUN python -m compileall -q app tests main.py

USER appuser

CMD ["python", "-m", "pytest", "-q"]
