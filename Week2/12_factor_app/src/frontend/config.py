# src/config.py
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parent
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}") 

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

VOCAB_PATH = MODELS_DIR / "vocab.pkl"
ENCODED_IMAGE_VAL_PATH = MODELS_DIR / "EncodedImageValResNet.pkl" # Used by backend models.py
# CAPTIONS_PATH = EXTERNAL_DATA_DIR / "captions.txt" # Used by backend main.py
# MODEL_PATH = MODELS_DIR / "BestModel.pth" # Used by backend models.py
# IMAGES_DIR = RAW_DATA_DIR / "Images" # Used by frontend app.py

CAPTIONS_PATH = "/home/ronak/fuse_class/Week2/12_factor_app/data/external/captions.txt"
IMAGES_DIR = "/home/ronak/fuse_class/Week2/12_factor_app/data/raw/Images"