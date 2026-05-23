# Chinese-CLIP + FAISS + Gradio 图文检索系统

这是一个最小可用的图文检索 Demo，使用 Chinese-CLIP 提取文本和图片 embedding，使用 FAISS `IndexFlatIP` 做向量检索，并用 Gradio 提供 Web UI。

## 功能列表

- 文搜图：输入中文描述，返回相似图片。
- 图搜图：上传查询图片，返回相似图片。
- 离线建库：批量编码 `data/images` 中的图片。
- 本地索引：保存 `image_embeddings.npy`、`image_paths.json` 和 `faiss.index`。

## 环境安装

推荐 Python 3.10。

```bash
conda create -n chinese_clip_retrieval python=3.10 -y
conda activate chinese_clip_retrieval
pip install -r requirements.txt
```

## 图片数据准备

把待检索图片放入：

```text
data/images/
```

支持子目录和以下格式：

```text
.jpg .jpeg .png .bmp .webp
```

也可以直接准备 Flickr30K 全量数据。脚本会从 Hugging Face 的 `AnyModal/flickr30k` 流式读取 `train`、`validation`、`test` 三个 split，把图片保存到 `data/images/flickr30k/`，并为每个 split 保存 `metadata.jsonl`：

```bash
python data/prepare_flickr30k.py
```

如果只想快速下载小样本，可以使用：

```bash
python data/prepare_flickr30k.py --limit 100
```

## 建立索引

一键完成图片编码和 FAISS 建库：

```bash
python scripts/build_all.py
```

成功后会生成：

```text
data/embeddings/image_embeddings.npy
data/embeddings/image_paths.json
data/embeddings/faiss.index
```

## 启动 UI

```bash
python run_app.py
```

打开 Gradio 输出的本地链接即可使用。

## 文搜图示例

可以在 Text Search 中输入：

```text
一只猫坐在沙发上
```

也可以用命令行测试：

```bash
python scripts/test_search.py --text "一只猫" --top_k 5
```

## 图搜图示例

在 Image Search 中上传一张图片，点击搜索后会返回图片库中最相似的 Top-K 图片。

## 常见问题

### data/images 中没有图片

请先把图片放入 `data/images/`，然后运行：

```bash
python scripts/build_all.py
```

### 未找到 FAISS index

请先运行：

```bash
python scripts/build_all.py
```

### Gradio 页面启动慢

UI 启动时会加载 Chinese-CLIP 模型、FAISS index 和图片路径，第一次启动需要等待模型加载完成。

## 后续升级方向

- 将 FAISS 替换为 Qdrant 或 Milvus。
- 增加 metadata filter 和在线增删图片。
- 更换更强的多模态 embedding 模型。
- 加入 reranker 做两阶段检索。
- 扩展为多模态 RAG，支持 OCR、PDF 页面截图和图表检索。
