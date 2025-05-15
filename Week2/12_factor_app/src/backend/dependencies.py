from pydantic import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Image Captioning API"
    model_path: str = "/app/models/BestModel.pth"
    vocab_path: str = "/app/models/vocab.pkl"
    embeddings_path: str = "/app/models/EncodedImageValResNet.pkl"
    captions_path: str = "/app/data/external/captions.txt"
    images_path: str = "/app/data/raw/Images"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()