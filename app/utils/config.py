"""项目全局配置。"""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "data" / "images"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDING_PATH = EMBEDDING_DIR / "image_embeddings.npy"
IMAGE_PATHS_PATH = EMBEDDING_DIR / "image_paths.json"
FAISS_INDEX_PATH = EMBEDDING_DIR / "faiss.index"

MODEL_NAME = "OFA-Sys/chinese-clip-vit-base-patch16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
TOP_K = 8

