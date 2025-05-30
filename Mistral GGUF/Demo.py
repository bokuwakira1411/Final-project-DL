import streamlit as st
from streamlit import session_state

from Main import Main
from GGUFClient import GGUFClient

# Load GGUF model
client = GGUFClient("/content/mistral.Q4_K_M.gguf")

# Mapping tên task hiển thị → name_task chính xác
task_map = {
    "classification": "classification",
    "simple qa": "qa_knowledge",
    "reasoning (social text)": "reasoning",
    "math": "computation"
}

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

st.title("🤖 Chat with Mistral")

# Lưu lịch sử chat
if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.text_area("Nhập câu hỏi hoặc nội dung:", height=100)

if st.button("Enter") and user_input.strip():
    # Lấy đúng task name
    name_task = task_map[selected_task]

    # Tạo task handler mới theo lựa chọn
    task_handler = Main(client, name_task=name_task)

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
