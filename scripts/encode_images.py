"""批量编码图片并保存 image embeddings。"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.chinese_clip_model import ChineseCLIPModelWrapper
from app.utils import config
from app.utils.image_utils import list_image_files, load_image


def chunk_list(items: list[str], batch_size: int) -> list[list[str]]:
    """把列表按 batch_size 切分。"""
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]


def encode_images() -> None:
    """读取 data/images，编码图片并保存 embedding 和路径映射。"""
    image_paths = list_image_files(config.IMAGE_DIR)
    if not image_paths:
        raise RuntimeError("data/images 中没有图片，请先添加图片。")

    config.EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    model = ChineseCLIPModelWrapper(config.MODEL_NAME, config.DEVICE)

    all_embeddings: list[np.ndarray] = []
    valid_paths: list[str] = []
    failed_count = 0

    for batch_paths in tqdm(chunk_list(image_paths, config.BATCH_SIZE), desc="Encoding images"):
        images = []
        loaded_paths = []
        for image_path in batch_paths:
            try:
                images.append(load_image(image_path))
                loaded_paths.append(image_path)
            except Exception as error:
                failed_count += 1
                print(f"warning: 跳过坏图 {image_path}: {error}")

        if not images:
            continue

        embeddings = model.encode_images(images)
        all_embeddings.append(embeddings)
        valid_paths.extend(loaded_paths)

    if not all_embeddings:
        raise RuntimeError("没有图片成功编码，请检查 data/images 中的图片文件。")

    image_embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    relative_paths = [
        str(Path(path).resolve().relative_to(config.PROJECT_ROOT))
        for path in valid_paths
    ]

    np.save(config.EMBEDDING_PATH, image_embeddings)
    with config.IMAGE_PATHS_PATH.open("w", encoding="utf-8") as file:
        json.dump(relative_paths, file, ensure_ascii=False, indent=2)

    print(f"总图片数: {len(image_paths)}")
    print(f"成功编码数: {len(valid_paths)}")
    print(f"失败图片数: {failed_count}")
    print(f"保存 embeddings: {config.EMBEDDING_PATH}")
    print(f"保存图片路径: {config.IMAGE_PATHS_PATH}")


if __name__ == "__main__":
    encode_images()

