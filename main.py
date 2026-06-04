import requests
import base64
from io import BytesIO
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from PIL import Image

# Initialize FastAPI
app = FastAPI(
    title="Jina CLIP v2 API",
    description="Local API for multimodal embeddings via jina-clip-v2."
)

# Load the model
print("Loading jina-clip-v2 model...")
model = SentenceTransformer('jinaai/jina-clip-v2', trust_remote_code=True)
print(f"Model loaded. Using device: {model.device}")

# ---------------------------------------------------------
# API Schema
# ---------------------------------------------------------
class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = Field(default_factory=list, description="List of text strings to embed.")
    image_urls: Optional[List[str]] = Field(default_factory=list, description="List of image URLs to embed.")
    images_base64: Optional[List[str]] = Field(default_factory=list, description="List of Base64 encoded image strings.")
    task: Optional[Literal["retrieval", "text-matching", "code"]] = Field(default=None, description="Unused, kept for API compatibility.")
    prompt_name: Optional[Literal["query", "passage"]] = Field(default=None, description="Unused, kept for API compatibility.")

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str = "jina-clip-v2"

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url, stream=True, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image from URL ({url}): {str(e)}")

def load_image_from_base64(b64_string: str) -> Image.Image:
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]
        image_data = base64.b64decode(b64_string)
        return Image.open(BytesIO(image_data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

# ---------------------------------------------------------
# Main Endpoint
# ---------------------------------------------------------
@app.post("/api/embed", response_model=EmbedResponse)
async def create_embeddings(req: EmbedRequest):
    if not any([req.texts, req.image_urls, req.images_base64]):
        raise HTTPException(status_code=400, detail="You must provide texts, image_urls, or images_base64.")

    unified_embeddings = []

    # 1. Process texts
    if req.texts:
        text_embs = model.encode(req.texts).tolist()
        unified_embeddings.extend(text_embs)

    # 2. Gather all images (URLs + base64)
    images_to_process = []

    if req.image_urls:
        images_to_process.extend([load_image_from_url(url) for url in req.image_urls])

    if req.images_base64:
        images_to_process.extend([load_image_from_base64(b64) for b64 in req.images_base64])

    # 3. Process images
    if images_to_process:
        image_embs = model.encode(images_to_process).tolist()
        unified_embeddings.extend(image_embs)

    return {
        "embeddings": unified_embeddings,
        "model": "jina-clip-v2"
    }
