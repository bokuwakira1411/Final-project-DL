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
# comp_model = SentenceTransformer('all-MiniLM-L6-V2')
# model.save('models/sbert')  # Save về local
with open('D:/test/Final Project DL/Data/classification_instruction_data_en.jsonl','r', encoding='utf-8') as f:
    classification_data =[json.loads(line) for line in f]
main = Main.Main(tokenizer, model, 'classification')
print(main.main('I hate UET', name_prompt='Few-shots ToT expanded',
                do_print=True))
