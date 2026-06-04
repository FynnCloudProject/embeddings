import os


class Settings:
    MODEL_NAME: str = os.environ.get("MODEL_NAME", "jinaai/jina-clip-v2")
    HOST: str = os.environ.get("HOST", "0.0.0.0")
    PORT: int = int(os.environ.get("PORT", "8000"))
    WORKERS: int = int(os.environ.get("WORKERS", "1"))
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "info")


settings = Settings()
