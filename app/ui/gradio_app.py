"""Gradio 图文检索界面。"""

import gradio as gr

from app.retrieval.search_engine import SearchEngine
from app.utils import config


def format_results(results: list[dict]) -> tuple[list[tuple[str, str]], list[list]]:
    """把检索结果转换为 Gallery 和 Dataframe 需要的格式。"""
    gallery = [
        (result["path"], f"score={result['score']:.4f}")
        for result in results
    ]
    table = [
        [result["rank"], result["path"], round(result["score"], 4)]
        for result in results
    ]
    return gallery, table


def create_app() -> gr.Blocks:
    """创建 Gradio 应用。"""
    try:
        engine = SearchEngine()
        startup_error = ""
    except Exception as error:
        engine = None
        startup_error = str(error)

    def text_search(query: str, top_k: int) -> tuple[list[tuple[str, str]], list[list]]:
        """处理文搜图按钮回调。"""
        if engine is None:
            raise gr.Error(startup_error)
        results = engine.search_by_text(query, int(top_k))
        return format_results(results)

    def image_search(image, top_k: int) -> tuple[list[tuple[str, str]], list[list]]:
        """处理图搜图按钮回调。"""
        if engine is None:
            raise gr.Error(startup_error)
        results = engine.search_by_image(image, int(top_k))
        return format_results(results)

    with gr.Blocks(title="Chinese-CLIP 图文检索 Demo") as demo:
        gr.Markdown("# Chinese-CLIP 图文检索 Demo")
        if startup_error:
            gr.Markdown(f"**启动提示：** {startup_error}")

        with gr.Tab("Text Search"):
            text_input = gr.Textbox(label="中文描述", placeholder="例如：一只猫坐在沙发上")
            text_top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=config.TOP_K,
                step=1,
                label="Top K",
            )
            text_button = gr.Button("搜索")
            text_gallery = gr.Gallery(label="图片结果", columns=5, rows=2, height=520, object_fit="cover")
            text_table = gr.Dataframe(headers=["rank", "path", "score"], label="检索结果")
            text_button.click(
                text_search,
                inputs=[text_input, text_top_k],
                outputs=[text_gallery, text_table],
            )

        with gr.Tab("Image Search"):
            image_input = gr.Image(label="查询图片", type="pil")
            image_top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=config.TOP_K,
                step=1,
                label="Top K",
            )
            image_button = gr.Button("搜索")
            image_gallery = gr.Gallery(label="图片结果", columns=5, rows=2, height=520, object_fit="cover")
            image_table = gr.Dataframe(headers=["rank", "path", "score"], label="检索结果")
            image_button.click(
                image_search,
                inputs=[image_input, image_top_k],
                outputs=[image_gallery, image_table],
            )

    return demo


def main() -> None:
    """启动 Gradio UI。"""
    demo = create_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)

