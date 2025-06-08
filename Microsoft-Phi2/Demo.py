import streamlit as st
import gc
import os
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from Main import Main
gc.collect()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

st.set_page_config(page_title='Microsoft-Phi2 Prompt Playground', layout='wide')

task_map = {
    "classification": "classification",
    "simple qa": "qa_knowledge",
    "reasoning (social text)": "reasoning",
    "math": "computation",
    "general": "general"
}

with st.sidebar:
    st.title("⚙️ Cấu hình")
    selected_task = st.selectbox("Select task", list(task_map.keys()), key="task_selector")
    prompt_type = st.selectbox("Select prompt type", [
        "Direct zero-shot", "Zero-shot CoT",
        "Zero-shot CoT + Self-consistency", "Zero-shot ToT",
        "Zero-shot ToT expanded", "Direct few-shots",
        "Few-shots CoT", "Few-shots CoT + Self-consistency",
        "Few-shots ToT", "Few-shots ToT expanded",
        "Few-shots CoT + ART"
    ])
    show_prompt = st.checkbox("Hiển thị prompt ?", value=True)

st.title("🤖 Chat with Phi2")

if (
    "model" not in st.session_state
    or "tokenizer" not in st.session_state
    or st.session_state.get("last_task") != selected_task
):
    with st.spinner("Đang tải lại mô hình..."):
        model_id = "microsoft/phi-2"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
                
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16,  
            low_cpu_mem_usage=True
        ).eval().to('cuda')

        st.session_state.tokenizer = tokenizer
        st.session_state.model = model
        st.session_state.last_task = selected_task

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_area("Nhập câu hỏi hoặc nội dung:", height=100)

if st.button("Enter") and user_input.strip():
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    name_task = task_map[selected_task]
    task_handler = Main(
        tokenizer=tokenizer,
        model=model,
        name_task=name_task
    )

    with st.spinner("Đang xử lý..."):
        response = task_handler.main(
            user_input,
            name_prompt=prompt_type,
            do_print=show_prompt
        )

    if prompt_type in ["Zero-shot CoT + Self-consistency", "Few-shots CoT + Self-consistency"] and isinstance(response, list):
        content = ""
        for idx, sample in enumerate(response):
            content += f"**Sample {idx+1}:**\n```\n{sample}\n```\n\n"
        st.session_state.history.append({"role": "assistant", "content": content})
    else:
        st.session_state.history.append({"role": "assistant", "content": response})

if st.session_state.history:
    for msg in st.session_state.history[-2:]:  # chỉ hiển thị 1 vòng tương tác
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
