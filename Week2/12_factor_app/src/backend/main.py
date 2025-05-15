# src/backend/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import string

# Relative import to models.py and config.py
from .model import load_model_and_vocab, generate_caption

from . import config
app = FastAPI(title="Image Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501/", "http://frontend:8501/", ""],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)

# Load model and vocab at startup using paths from config
model, word_to_index, index_to_word, vocab_size = load_model_and_vocab()

# Pydantic model for request
class ImageRequest(BaseModel):
    image_name: str

# Load unique validation image names using CAPTIONS_PATH from config
try:
    df = pd.read_csv(config.CAPTIONS_PATH)
    # Assuming the same 90/10 split for validation images as in your prototype
    val_df = df.iloc[int(0.9 * len(df)):]
    # Ensure 'image' column exists
    if 'image' not in val_df.columns:
        raise RuntimeError("Column 'image' not found in captions file.")
    unique_validation_image_names = val_df[['image']].drop_duplicates()['image'].tolist()
except FileNotFoundError:
    config.logger.error(f"Captions file not found at: {config.CAPTIONS_PATH}")
    unique_validation_image_names = [] # Fallback to empty list
except Exception as e:
    config.logger.error(f"Error loading captions or determining unique images: {e}")
    unique_validation_image_names = []


@app.on_event("startup")
async def startup_event():
    if not unique_validation_image_names:
        config.logger.warning("Validation image list is empty. Check captions file and path.")
    else:
        config.logger.info(f"Loaded {len(unique_validation_image_names)} unique validation image names.")


@app.get("/images")
async def get_image_list():
    """Returns a list of available validation image names."""
    if not unique_validation_image_names:
         raise HTTPException(status_code=404, detail="No validation images available. Check server configuration.")
    return {"images": unique_validation_image_names}


@app.post("/predict")
async def predict_caption(request: ImageRequest):
    """Generates a caption for a given image name."""
    if request.image_name not in unique_validation_image_names:
        raise HTTPException(status_code=404, detail=f"Image '{request.image_name}' not found in the available validation set.")

    try:
        predicted_caption = generate_caption(model, word_to_index, index_to_word, vocab_size, request.image_name)
        
        # Fetch actual captions for the image
        actual_captions = val_df[val_df['image'] == request.image_name]['caption'].tolist()
        
        return {
            "image_name": request.image_name,
            "predicted_caption": predicted_caption,
            "actual_captions": actual_captions
        }
    except ValueError as e:
        # Catch specific error from generate_caption if image embedding is missing
        if "not found in embeddings" in e(e):
            config.logger.error(f"Embedding not found for {request.image_name}: {e}")
            raise HTTPException(status_code=404, detail=str(e))
        config.logger.error(f"ValueError during caption generation for {request.image_name}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    except Exception as e:
        config.logger.error(f"Unexpected error during prediction for {request.image_name}: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")