from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from basic_prompt_designs.Classfication import Classification


class Main:
    def __init__(self, tokenizer, model, name_task, type_prompt=None):
        self.name_task = name_task
        self.type_prompt = type_prompt
        if name_task=='classification':
            self.task = Classification(tokenizer,model)

    def main(self, text, name_prompt, do_print=False):
        return self.task.run(text, do_print, name_prompt)