FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=10000

WORKDIR /app/CICTify

# System deps for Pillow / PDF rendering / browser-backed tooling.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libglib2.0-0 \
        libnss3 \
        libgconf-2-4 \
        libxss1 \
        libasound2 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcups2 \
        libdrm2 \
        libxkbcommon0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libgbm1 \
        libgtk-3-0 \
        libpango-1.0-0 \
        libcairo2 \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

COPY CICTify/requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt \
    && python -m playwright install --with-deps chromium

COPY CICTify/ ./

EXPOSE 10000

# Single worker keeps the model/runtime memory footprint predictable.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 superai_web:app"]