#!/bin/sh
set -eu

cd "$(dirname "$0")/../.."
if [ ! -f .env.aws ]; then
  echo "Missing .env.aws"
  exit 1
fi

set -a
. ./.env.aws
set +a

mkdir -p backups
timestamp=$(date -u +%Y%m%dT%H%M%SZ)
backup_path="backups/admissions-${timestamp}.sql"

docker compose --env-file .env.aws -f compose.aws.yaml exec -T db \
  pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" > "$backup_path"

echo "Database backup written to $backup_path"
