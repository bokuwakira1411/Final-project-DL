# google/flan-t5-xl
#C:\Users\Admin\anaconda3\python.exe -m streamlit run Demo.py

import streamlit as st
from streamlit import session_state
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import Main
from basic_prompt_designs.Classfication import Classification

# model_name = "google/flan-t5-base"
save_path = "C:/Users/Admin/PycharmProjects/Final-project-DL/models/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(save_path)

model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
model.to('cuda')
# # comp_model = SentenceTransformer('models/sbert')
# # comp_model.save('models/sbert')  # Save về local
#
# # with open('D:/test/Final Project DL/Data/classification_instruction_data_en.jsonl','r', encoding='utf-8') as f:
# #     classification_data =[json.loads(line) for line in f]
# # with open('C:/Users/Admin/PycharmProjects/Final-project-DL/Data/final_train_math.json','r', encoding='utf-8') as f:
# #     computation_data = json.load(f)
# # texts = [task['input'] for task in computation_data]
# # vectorizer = TfidfVectorizer().fit(texts)
# # X = vectorizer.transform(texts)
st.set_page_config(page_title='Flan T5 XL Prompt Playground', layout='wide')
main = None
with st.sidebar:
    st.title('Cấu hình')
    task = st.selectbox('Select task', ['classification', 'simple qa', 'reasoning',
                                        'computation'])
    main = Main.Main(tokenizer, model, task)
    prompt_type = st.selectbox('Select prompt type', ['Direct zero-shot', 'Zero-shot CoT',
                                                    'Zero-shot CoT + Self-consistency', 'Zero-shot CoT + ART',
                                                    'Zero-shot ToT', 'Zero-shot ToT expanded',
                                                    'Direct few-shots', 'Few-shots CoT',
                                                    'Few-shots CoT + Self-consistency', 'Few-shots CoT + ART',
                                                                                        'Few-shots ToT',
                                                    'Few-shots ToT expanded',
                                                    ])
    show_prompt = st.checkbox('Print prompt ?', value=False)
    send = st.button('OK')
st.title('Chat with Flan-T5-XL')
if 'history' not in session_state:
    st.session_state['history'] = []
user_input = st.text_area('Enter your question')
if st.button('Enter') and user_input.strip():
    response = main.main(user_input, name_prompt=prompt_type, do_print=show_prompt)
    st.session_state.history.append({'role': 'user', 'content': user_input})
    st.session_state.history.append({'role': 'assistant', 'content': response})

for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])