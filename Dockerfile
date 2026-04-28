FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

WORKDIR /app/CICTify

# System deps for PDF rendering only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libnss3 \
        libcairo2 \
    && rm -rf /var/lib/apt/lists/*

COPY CICTify/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY CICTify/ ./

EXPOSE 10000

# Single worker keeps the model/runtime memory footprint predictable.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 superai_web:app"]