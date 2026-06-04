import asyncio

from fastapi import APIRouter, HTTPException

from app.embedding import embedding_model, load_image_from_url, load_image_from_base64
from app.schemas import EmbedRequest, EmbedResponse

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "model": embedding_model.model.model_card_data.model_id}


@router.post("/api/embed", response_model=EmbedResponse)
async def create_embeddings(req: EmbedRequest):
    if not any([req.texts, req.image_urls, req.images_base64]):
        raise HTTPException(status_code=400, detail="Provide at least one of: texts, image_urls, images_base64.")

    unified_embeddings = []

    if req.texts:
        text_embeddings = await asyncio.to_thread(embedding_model.encode_texts, req.texts)
        unified_embeddings.extend(text_embeddings)

    images_to_process = []

    if req.image_urls:
        for url in req.image_urls:
            try:
                images_to_process.append(load_image_from_url(url))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load image from URL ({url}): {e}")

    if req.images_base64:
        for b64 in req.images_base64:
            try:
                images_to_process.append(load_image_from_base64(b64))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}")

    if images_to_process:
        image_embeddings = await asyncio.to_thread(embedding_model.encode_images, images_to_process)
        unified_embeddings.extend(image_embeddings)

    return EmbedResponse(
        embeddings=unified_embeddings,
        model=embedding_model.model.model_card_data.model_id,
        dimensions=embedding_model.dimensions,
    )
