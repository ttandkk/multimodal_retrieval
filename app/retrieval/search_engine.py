"""图文检索核心业务逻辑。"""

import json
from pathlib import Path

from PIL import Image

from app.models.chinese_clip_model import ChineseCLIPModelWrapper
from app.retrieval.faiss_index import FaissIndex
from app.utils import config


class SearchEngine:
    """加载模型、图片路径和 FAISS 索引，并提供检索接口。"""

    def __init__(self) -> None:
        """启动时一次性加载检索所需资源。"""
        self.model = ChineseCLIPModelWrapper(config.MODEL_NAME, config.DEVICE)
        self.index = FaissIndex(config.FAISS_INDEX_PATH)
        self.image_paths = self._load_image_paths(config.IMAGE_PATHS_PATH)

    def search_by_text(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """根据中文文本检索图片。"""
        if not query.strip():
            raise ValueError("请输入查询文本。")
        query_embedding = self.model.encode_text([query])
        return self._search(query_embedding, top_k)

    def search_by_image(self, image: Image.Image | None, top_k: int = config.TOP_K) -> list[dict]:
        """根据上传图片检索相似图片。"""
        if image is None:
            raise ValueError("请上传查询图片。")
        query_embedding = self.model.encode_images([image])
        return self._search(query_embedding, top_k)

    def _search(self, query_embedding, top_k: int) -> list[dict]:
        """执行 FAISS 检索并整理返回结果。"""
        if not self.image_paths:
            return []

        real_top_k = min(int(top_k), len(self.image_paths))
        scores, indices = self.index.search(query_embedding, real_top_k)

        results: list[dict] = []
        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "path": self.image_paths[int(index)],
                    "score": float(score),
                }
            )
        return results

    def _load_image_paths(self, json_path: str | Path) -> list[str]:
        """读取图片路径列表。"""
        path_file = Path(json_path)
        if not path_file.exists():
            raise FileNotFoundError(
                "未找到 image_paths.json，请先运行 python scripts/build_all.py"
            )
        with path_file.open("r", encoding="utf-8") as file:
            image_paths: list[str] = json.load(file)
        return image_paths

