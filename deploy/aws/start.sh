#!/bin/sh
set -eu

cd "$(dirname "$0")/../.."

if [ ! -f .env.aws ]; then
  echo "Missing .env.aws. Copy deploy/aws/aws.env.example to .env.aws and replace every REPLACE_WITH value."
  exit 1
fi

if grep -q "REPLACE_WITH" .env.aws; then
  echo ".env.aws still contains placeholder values. Update it before deployment."
  exit 1
fi

docker compose --env-file .env.aws -f compose.aws.yaml up -d --build
docker compose --env-file .env.aws -f compose.aws.yaml ps

echo "The first startup can take several minutes while Qwen and the embedding model download."
echo "Watch progress with: docker compose --env-file .env.aws -f compose.aws.yaml logs -f"
