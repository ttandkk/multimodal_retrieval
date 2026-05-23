"""Chinese-CLIP 文本和图像向量编码封装。"""

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import ChineseCLIPModel as HFChineseCLIPModel
from transformers import ChineseCLIPProcessor


class ChineseCLIPModelWrapper:
    """使用 Hugging Face Chinese-CLIP 提取归一化 embedding。"""

    def __init__(self, model_name: str, device: str) -> None:
        """加载 Chinese-CLIP processor 和模型。"""
        self.device = torch.device(device)
        try:
            self.processor = ChineseCLIPProcessor.from_pretrained(model_name)
            self.model = HFChineseCLIPModel.from_pretrained(model_name).to(self.device)
        except Exception as error:
            raise RuntimeError(
                "Chinese-CLIP 模型加载失败，请检查网络、模型名或本地缓存。"
            ) from error
        self.model.eval()

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为 L2 归一化的 float32 向量。"""
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = self._move_inputs_to_device(inputs)
        features = self.model.get_text_features(**inputs)
        embeddings = self._features_to_numpy(features)
        return self.normalize(embeddings)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """将 PIL 图片列表编码为 L2 归一化的 float32 向量。"""
        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.processor(images=rgb_images, return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)
        features = self.model.get_image_features(**inputs)
        embeddings = self._features_to_numpy(features)
        return self.normalize(embeddings)

    def normalize(self, embeddings: np.ndarray) -> np.ndarray:
        """对 embedding 做 L2 normalize，并转换为 np.float32。"""
        embeddings = embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        return (embeddings / np.maximum(norms, 1e-12)).astype(np.float32)

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        """将 processor 输出移动到模型所在设备。"""
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _features_to_numpy(self, features: Any) -> np.ndarray:
        """兼容不同 transformers 版本的特征输出格式。"""
        if isinstance(features, torch.Tensor):
            tensor = features
        elif hasattr(features, "pooler_output"):
            tensor = features.pooler_output
        else:
            tensor = features[1]
        return tensor.detach().cpu().numpy()

