"""命令行检索测试脚本。"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.search_engine import SearchEngine


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="测试 Chinese-CLIP 图文检索")
    parser.add_argument("--text", type=str, default="一只猫", help="中文查询文本")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数量")
    return parser.parse_args()


def main() -> None:
    """执行一次文搜图测试并打印结果。"""
    args = parse_args()
    engine = SearchEngine()
    results = engine.search_by_text(args.text, args.top_k)
    for result in results:
        print(f"{result['rank']}. {result['path']} score={result['score']:.4f}")


if __name__ == "__main__":
    main()

