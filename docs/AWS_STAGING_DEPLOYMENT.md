# AWS staging deployment

This staging layout runs the complete assistant on one ARM-based EC2 instance.
Only SSH and the Nginx web port are public. PostgreSQL, Ollama, FastAPI and SMTP
remain on a private Docker network. Mailpit binds only to the EC2 loopback
interface and is viewed through an SSH tunnel.

## Local validation

From the repository root:

```bash
cp deploy/aws/aws.env.example .env.aws
```

Replace all placeholder values. Generate secrets with:

```bash
openssl rand -hex 32
```

Use the same generated database password in `POSTGRES_PASSWORD` and inside
`DATABASE_URL`. For a local production-layout test, set:

```text
ALLOWED_ORIGINS=http://localhost
```

Start the stack:

```bash
./deploy/aws/start.sh
```

Check:

```bash
curl http://localhost/health
curl http://localhost/health/model
```

Open the chatbot at `http://localhost`, the admin portal at
`http://localhost/admin`, and Mailpit at `http://localhost:8025`.

Stop without deleting data:

```bash
docker compose --env-file .env.aws -f compose.aws.yaml down
```

Never add `-v` unless the database and Ollama volumes are intentionally being
deleted.

## EC2 target

- Region: `ap-south-1` (Mumbai)
- Architecture: ARM64
- Operating system: Ubuntu Server 24.04 LTS
- Initial size: `t4g.xlarge` when permitted by the AWS Free Plan
- Storage: 40 GiB gp3 EBS
- Public inbound ports: 22 from the administrator IP and 80 from test users

Do not expose ports 5432, 8000, 8025, 1025 or 11434 in the EC2 security group.

## Mailpit access from the administrator Mac

With the SSH session available:

```bash
ssh -i /path/to/key.pem -L 8025:127.0.0.1:8025 ubuntu@EC2_PUBLIC_IP
```

Then open `http://localhost:8025`. The Mailpit UI is not public.

## Persistence and backups

Named Docker volumes preserve PostgreSQL data, Ollama models and the Hugging
Face embedding cache across container restarts. EBS preserves those volumes
while the EC2 instance is stopped.

Create a logical database backup with:

```bash
./deploy/aws/backup-db.sh
```

The generated `backups/` directory is ignored by Git and must never be pushed.
