# google/flan-t5-xl
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

import Main
from basic_prompt_designs.Classfication import Classification

# model_name = "google/flan-t5-base"
save_path = "models/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(save_path)

model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
model.to('cuda')
# comp_model = SentenceTransformer('models/sbert')
# comp_model.save('models/sbert')  # Save về local

# with open('D:/test/Final Project DL/Data/classification_instruction_data_en.jsonl','r', encoding='utf-8') as f:
#     classification_data =[json.loads(line) for line in f]
# with open('C:/Users/Admin/PycharmProjects/Final-project-DL/Data/final_train_math.json','r', encoding='utf-8') as f:
#     computation_data = json.load(f)
# texts = [task['input'] for task in computation_data]
# vectorizer = TfidfVectorizer().fit(texts)
# X = vectorizer.transform(texts)
main = Main.Main(tokenizer, model, 'reasoning')
text =  "Act as a software developer , I would like to grab a page from a pdf report and attach it in email body how do I go about it?",
print(main.main(f'{text}', name_prompt='Zero-shot CoT',
                do_print=True))
# main = Main.Main(tokenizer, model, 'classification')
# print(main.main('can you help me to solve this task', name_prompt='Few-shots CoT',
#                 do_print=True))
