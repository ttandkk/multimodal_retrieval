"""预处理辅助函数。"""

from PIL import Image


def ensure_rgb_image(image: Image.Image) -> Image.Image:
    """确保输入图片为 RGB 格式。"""
    return image.convert("RGB")

