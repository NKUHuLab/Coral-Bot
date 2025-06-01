# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 15:27:16 2025

@author: Deft
"""

import gradio as gr
# rag_modul import rag_pipeline
from rag_module_adjust import rag_pipeline, load_embeddings_and_index

def generate_response(query):
    folder_path = "G:/LLMs coral/LLMs coral bot/"
    model_path = "G:/LLMs coral/Embedding/bge-m3"
    all_chunks, index, chunk_to_file_mapping = load_embeddings_and_index(
        embeddings_path=f"{folder_path}/embeddings.npy", 
        index_path=f"{folder_path}/faiss_index.index",
        chunks_path=f"{folder_path}/text_chunks.json"
    )
    #answer, references = rag_pipeline(query, folder_path, model_path)
    answer, references = rag_pipeline(query, folder_path, model_path, all_chunks=all_chunks, index=index, chunk_to_file_mapping=chunk_to_file_mapping)
    return answer, references

# 自定义 CSS 设置提交按钮颜色和标题、参考文本样式
custom_css = """
.logo {display: flex; align-items: center; justify-content: center;}
#submit-btn {
    background: #4CAF50 !important;  /* 绿色 */
    color: white !important;
    transition: background 0.3s ease;  /* 过渡效果 */
}
#submit-btn:hover {
    background: #45a049 !important;  /* 悬停时深绿色 */
}
#submit-btn:active {
    background: #388E3C !important;  /* 点击时更深绿色 */
}
#title {
    text-align: center;
    font-size: 36px;  /* 标题字号 */
    font-weight: bold;
}
"""

# 使用 Blocks 自定义布局
with gr.Blocks(title="Chat with coral policy bot!", css=custom_css, analytics_enabled=False) as ui:
    # 图片和标题行
    with gr.Row(elem_classes="logo"):
        gr.Image("G:/LLMs coral/LLMs coral bot/coral1.png", width=40, show_label=False)
    
    # 标题部分 - 使用 Markdown 并将其居中
    gr.Markdown("<div style='text-align: center; font-size: 36px; font-weight: bold;'>Chat with coral policy bot!</div>")
    
    # 输入输出组件
    input_box = gr.Textbox(lines=15, placeholder="Please enter your questions", label="Enter your questions")
    examples = gr.Examples(
        examples=[["What is coral reef?"],
                  ["How can we protect the coral reefs?"], 
                  ["What policy recommendations can you provide for coral reef conservation?"]], 
        inputs=[input_box],
        label="Example questions"
    )
    
    with gr.Row():
        answer_box = gr.Textbox(label="Answer", lines=20)
        ref_box = gr.Textbox(label="References", lines=20)
    
    # 提交和清除按钮
    with gr.Row():
        submit_btn = gr.Button("Submit", elem_id="submit-btn")  # 设置按钮的 ID
        clear_btn = gr.Button("Clear")
    
    # 提交按钮的功能
    submit_btn.click(
        fn=generate_response,
        inputs=input_box,
        outputs=[answer_box, ref_box]
    )
    
    # 清除按钮的功能
    clear_btn.click(
        fn=lambda: [None, None, None],  # 清空输入框、答案框和引用框
        inputs=[],
        outputs=[input_box, answer_box, ref_box]
    )

# 启动时设置网页图标
ui.launch(favicon_path="G:/LLMs coral/LLMs coral bot/coral1.png")
