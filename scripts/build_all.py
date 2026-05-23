"""一键完成图片编码和 FAISS 建库。"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_index import build_index
from scripts.encode_images import encode_images


def main() -> None:
    """连续执行图片编码和 FAISS 建库。"""
    encode_images()
    build_index()


if __name__ == "__main__":
    main()

