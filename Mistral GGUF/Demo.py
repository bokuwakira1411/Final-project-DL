import streamlit as st
from streamlit import session_state
from Main import Main

task_map = {
    "classification": "classification",
    "simple qa": "qa_knowledge",
    "reasoning (social text)": "reasoning",
    "math": "computation"
}
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",  # Tự động phân bổ GPU/CPU
    low_cpu_mem_usage=True
).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

st.set_page_config(page_title='Flan/Mistral Prompt Playground', layout='wide')

with st.sidebar:
    st.title("⚙️ Cấu hình")
    selected_task = st.selectbox("Select task", list(task_map.keys()))
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

# Lưu lịch sử chat
if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_area("Nhập câu hỏi hoặc nội dung:", height=100)

if st.button("Enter") and user_input.strip():
    # Lấy đúng task name
    name_task = task_map[selected_task]

    # Tạo task handler mới theo lựa chọn
    task_handler = Main(tokenizer,model, name_task=name_task)

    with st.spinner("Đang xử lý..."):
        response = task_handler.main(
            user_input,
            name_prompt=prompt_type,
            do_print=show_prompt
        )

    st.session_state.history.append({"role": "user", "content": user_input})
    st.session_state.history.append({"role": "assistant", "content": response})

# Hiển thị lịch sử trò chuyện
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
