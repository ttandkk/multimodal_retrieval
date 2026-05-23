"""图片读取和遍历工具。"""

from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_image_files(image_dir: str | Path) -> list[str]:
    """递归遍历图片目录，返回所有支持格式的图片路径。"""
    image_root = Path(image_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"{image_root} 不存在，请先创建图片目录。")

    image_paths = [
        str(path)
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


def load_image(image_path: str | Path) -> Image.Image:
    """读取图片并转换为 RGB。"""
    return Image.open(image_path).convert("RGB")

