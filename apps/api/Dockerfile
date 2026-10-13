FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY apps/api/requirements.txt apps/api/requirements-embeddings.txt ./
# Install PyTorch from its CPU-only index first. Installing sentence-transformers
# directly from PyPI can otherwise resolve the Linux CUDA build and add several
# gigabytes of GPU libraries that this CPU deployment cannot use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-embeddings.txt

COPY apps/api/app ./app
COPY deploy/aws/api-entrypoint.sh /usr/local/bin/api-entrypoint
RUN chmod +x /usr/local/bin/api-entrypoint

EXPOSE 8000
ENTRYPOINT ["api-entrypoint"]
