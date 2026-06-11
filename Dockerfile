FROM python:3.12-slim

# xgboost needs libgomp1 at runtime; ca-certificates for HTTPS to Alpaca/yfinance
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so code changes don't bust the wheel cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user matching Ubuntu's default UID (Oracle Cloud's ubuntu user is 1000)
RUN useradd --create-home --uid 1000 --shell /bin/bash trader

COPY --chown=trader:trader . .

USER trader

# tini reaps zombies and forwards SIGTERM cleanly so the loop shuts down on `docker stop`
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "scripts/run_paper.py"]
