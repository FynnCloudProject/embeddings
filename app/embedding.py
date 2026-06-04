import base64
import logging
from io import BytesIO
from typing import List

import requests
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper around the SentenceTransformer CLIP model."""

    def __init__(self):
        self._model: SentenceTransformer | None = None

    def load(self):
        logger.info(f"Loading model: {settings.MODEL_NAME}")
        self._model = SentenceTransformer(settings.MODEL_NAME, trust_remote_code=True)
        logger.info(f"Model loaded on device: {self._model.device}")

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
        return self.model.encode(images).tolist()


# Singleton instance
embedding_model = EmbeddingModel()


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_image_from_base64(b64_string: str) -> Image.Image:
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_data = base64.b64decode(b64_string)
    return Image.open(BytesIO(image_data)).convert("RGB")
