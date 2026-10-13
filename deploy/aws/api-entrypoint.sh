#!/bin/sh
set -eu

python - <<'PY'
import time
import psycopg
from app.config import get_settings

url = get_settings().database_url.replace("postgresql+psycopg://", "postgresql://", 1)
for attempt in range(60):
    try:
        with psycopg.connect(url, connect_timeout=5):
            print("Database is ready.")
            break
    except psycopg.OperationalError:
        if attempt == 59:
            raise
        time.sleep(2)
PY

if [ "${RUN_SEED_INGESTION:-true}" = "true" ]; then
  echo "Ingesting approved school knowledge..."
  python -m app.scripts.ingest_seed
fi

exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --proxy-headers \
  --forwarded-allow-ips="*"
