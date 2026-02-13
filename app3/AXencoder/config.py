import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "preprocessing")

# --- Model IDs ---
# STT Model (User Requested)
STT_MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
# Text Encoder Model
TEXT_ENCODER_ID = "skt/A.X-Encoder-base"

# --- Hyperparameters ---
BATCH_SIZE = 32
MAX_SEQ_LEN = 50  # Max number of chunks to consider for context
LEARNING_RATE = 2e-5
EPOCHS = 10

# --- Optimization ---
USE_FLASH_ATTENTION = True
USE_AMP = True  # Automatic Mixed Precision

# --- Logging ---
WANDB_PROJECT = "phishing-detection-axencoder"

# --- Label Mapping ---
LABEL_MAP = {
    "Normal": 0,
    "Phishing": 1
}

# Detailed Category Mapping
CATEGORY_TO_LABEL = {
    "대출 사기형": 1,
    "바로 이 목소리": 1,
    "수사기관 사칭형": 1,
    "이체 출금 대출서비스": 0,
    "잔고 및 거래내역": 0,
    "Normal": 0,
    "Voice_Phishing": 1
}
