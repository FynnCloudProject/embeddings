FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Accept the build argument
ARG HF_TOKEN
# Set it as an environment variable during this RUN layer
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True)"

# Copy application code
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
