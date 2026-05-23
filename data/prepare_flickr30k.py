"""下载并整理 Hugging Face Flickr30K 数据集图片。"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "images" / "flickr30k"
DEFAULT_HF_HOME = Path("/projects/hdd/tmp/hf_cache")
DEFAULT_SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="准备 Flickr30K 图片数据")
    parser.add_argument(
        "--dataset",
        type=str,
        default="AnyModal/flickr30k",
        help="Hugging Face 数据集名称",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="图片输出目录",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="需要处理的数据集 split",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="每个 split 最多保存多少张图片，默认保存全部",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的图片文件",
    )
    return parser.parse_args()


def setup_hf_cache() -> None:
    """把 Hugging Face 缓存放到 HDD 临时目录。"""
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_HOME / "datasets"))


def get_caption(row: dict[str, Any]) -> str:
    """从数据行中提取第一条 caption。"""
    captions = row.get("alt_text") or row.get("original_alt_text") or []
    if isinstance(captions, list) and captions:
        return str(captions[0])
    if isinstance(captions, str):
        return captions
    return ""


def save_split(dataset_name: str, split: str, output_dir: Path, limit: int | None, overwrite: bool) -> int:
    """保存一个 split 的图片和 metadata。"""
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.jsonl"

    dataset = load_dataset(dataset_name, split=split, streaming=True)
    saved_count = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, row in enumerate(dataset):
            if limit is not None and saved_count >= limit:
                break

            filename = row.get("filename") or f"{split}_{index:06d}.jpg"
            image_name = f"{index:06d}_{Path(filename).name}"
            image_path = split_dir / image_name

            if overwrite or not image_path.exists():
                image = row["image"].convert("RGB")
                image.save(image_path, format="JPEG", quality=95)

            metadata = {
                "split": split,
                "index": index,
                "file": str(image_path.relative_to(PROJECT_ROOT)),
                "source_filename": filename,
                "caption": get_caption(row),
            }
            metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

            saved_count += 1
            if saved_count % 500 == 0:
                print(f"{split}: saved {saved_count} images")

    print(f"{split}: done, saved {saved_count} images")
    return saved_count


def main() -> None:
    """下载并整理 Flickr30K 数据集。"""
    args = parse_args()
    setup_hf_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_count = 0
    for split in args.splits:
        total_count += save_split(
            args.dataset,
            split,
            args.output_dir,
            args.limit,
            args.overwrite,
        )

    print(f"全部完成，共保存 {total_count} 张图片到 {args.output_dir}")


if __name__ == "__main__":
    main()
