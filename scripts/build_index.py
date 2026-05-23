"""从 image_embeddings.npy 构建 FAISS 索引。"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.faiss_index import FaissIndex
from app.utils import config


def build_index() -> None:
    """读取图片 embedding 并保存 FAISS IndexFlatIP。"""
    if not config.EMBEDDING_PATH.exists():
        raise FileNotFoundError("未找到 image_embeddings.npy，请先运行 python scripts/encode_images.py")

    embeddings = np.load(config.EMBEDDING_PATH).astype(np.float32)
    if embeddings.ndim != 2:
        raise ValueError("image_embeddings.npy 必须是二维数组，shape 为 [N, D]。")

    faiss_index = FaissIndex()
    faiss_index.build(embeddings)
    faiss_index.save(config.FAISS_INDEX_PATH)

    print(f"FAISS index 已保存: {config.FAISS_INDEX_PATH}")
    print(f"向量数量: {embeddings.shape[0]}")
    print(f"向量维度: {embeddings.shape[1]}")


if __name__ == "__main__":
    build_index()

