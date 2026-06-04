# FynnCloud-Embeddings

Multimodal embedding service for FynnCloud. Runs [jina-clip-v2](https://huggingface.co/jinaai/jina-clip-v2) behind a FastAPI server to generate 1024-dimensional vectors from text and images.

Used by the backend for semantic file search (pgvector cosine similarity).

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The model weights get downloaded on first launch (~2 GB).

## Docker

```bash
docker build -t fynncloud-embeddings .
docker run -p 8000:8000 fynncloud-embeddings
```

The Dockerfile bakes the model weights into the image so there's no download at runtime.

## API

### `POST /api/embed`

```json
{
  "texts": ["hello world"],
  "image_urls": ["https://example.com/photo.jpg"],
  "images_base64": ["iVBORw0KGgo..."]
}
```

All fields are optional, but at least one must be provided. Returns:

```json
{
  "embeddings": [[0.012, -0.034, ...]],
  "model": "jinaai/jina-clip-v2",
  "dimensions": 1024
}
```

### `GET /health`

Returns server status.

## Config

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `jinaai/jina-clip-v2` | HuggingFace model ID |
| `PORT` | `8000` | Server port |
| `WORKERS` | `1` | Uvicorn worker count (each loads a full model copy) |
| `LOG_LEVEL` | `info` | Logging level |
