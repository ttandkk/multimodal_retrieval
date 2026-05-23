"""FAISS IndexFlatIP 索引封装。"""

from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """负责 FAISS 索引的构建、保存、加载和检索。"""

    def __init__(self, index_path: str | Path | None = None) -> None:
        """可选地从文件加载索引。"""
        self.index: faiss.Index | None = None
        if index_path is not None:
            self.load(index_path)

    def build(self, embeddings: np.ndarray) -> None:
        """使用 float32 embedding 构建 IndexFlatIP。"""
        if embeddings.ndim != 2:
            raise ValueError("embeddings 必须是二维数组，shape 为 [N, D]。")
        embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def save(self, index_path: str | Path) -> None:
        """保存 FAISS 索引到文件。"""
        if self.index is None:
            raise RuntimeError("FAISS index 尚未构建，无法保存。")
        index_file = Path(index_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))

    def load(self, index_path: str | Path) -> None:
        """从文件加载 FAISS 索引。"""
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(
                "未找到 FAISS index，请先运行 python scripts/build_all.py"
            )
        self.index = faiss.read_index(str(index_file))

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        """检索 top-k 相似向量，返回 scores 和 indices。"""
        if self.index is None:
            raise RuntimeError("FAISS index 尚未加载。")
        query_embedding = query_embedding.astype(np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
            raise ValueError("query_embedding shape 必须为 [1, D]。")
        return self.index.search(query_embedding, top_k)

