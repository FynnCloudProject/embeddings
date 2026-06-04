from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    texts: Optional[List[str]] = Field(default_factory=list, description="List of text strings to embed.")
    image_urls: Optional[List[str]] = Field(default_factory=list, description="List of image URLs to embed.")
    images_base64: Optional[List[str]] = Field(default_factory=list, description="List of Base64 encoded image strings.")
    task: Optional[Literal["retrieval", "text-matching", "code"]] = Field(default=None, description="Reserved for future use.")
    prompt_name: Optional[Literal["query", "passage"]] = Field(default=None, description="Reserved for future use.")


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int
